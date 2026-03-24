"""
Data generation script for 2D viscous Burgers equation.

PDE:  u_t + u * u_x + u * u_y = nu * (u_xx + u_yy)
Domain: [0, 2pi)^2, periodic boundary conditions
Method: Pseudo-spectral with integrating-factor Euler time stepping.

Why pre-generate instead of on-the-fly:
  Solving a 64x64 2D Burgers trajectory to 100 timesteps takes ~3-10 s on CPU.
  Loading a pre-saved HDF5 slice takes ~microseconds.  Pre-generating ~600
  samples (100 samples x 6 viscosities) is a one-time cost (~30 min); it
  follows the same pattern as all other datasets in this repo (NS, Burgers 1D,
  SOMA) and avoids making the DataLoader the training bottleneck.

Output HDF5 layout (one file per viscosity nu):
  tensor       : float32 (N, T, H, W)  -- scalar field u(x,y,t)
  t-coordinate : float32 (T,)           -- time values
  x-coordinate : float32 (H,)           -- x grid (0 .. 2pi, H points)
  y-coordinate : float32 (W,)           -- y grid (0 .. 2pi, W points)

Usage:
  python utils/data/generate_burgers2d.py
  python utils/data/generate_burgers2d.py --out_dir /my/data/dir --H 128 --W 128

Default output paths mirror the 1D Burgers convention:
  /pscratch/sd/g/gzhao27/INR/data/2D_Burgers_Sols_Nu{nu}.hdf5
"""

import argparse
import os

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _wavenumbers(H: int, W: int):
    """Return wavenumber grids for 2-D rfft2 layout (H x W//2+1)."""
    kx = np.fft.rfftfreq(W, d=1.0 / W).astype(np.float64)   # shape (W//2+1,)
    ky = np.fft.fftfreq(H,  d=1.0 / H).astype(np.float64)   # shape (H,)
    KX, KY = np.meshgrid(kx, ky)                              # (H, W//2+1)
    return KX, KY


def _random_ic(H: int, W: int, n_modes: int = 8, rng: np.random.Generator = None) -> np.ndarray:
    """
    Random initial condition: superposition of low-wavenumber Fourier modes
    on the domain [0, 2pi)^2.  The amplitude decays as 1/(k^2+l^2) so
    higher modes contribute less energy (smooth IC).
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, H, endpoint=False)
    X, Y = np.meshgrid(x, y)
    u = np.zeros((H, W), dtype=np.float64)
    for k in range(1, n_modes + 1):
        for l in range(1, n_modes + 1):
            amp = rng.uniform(-1.0, 1.0) / (k ** 2 + l ** 2)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            u += amp * np.sin(k * X + l * Y + phase)
    return u


def solve_burgers2d(
    u0: np.ndarray,
    nu: float,
    dt: float,
    T_steps: int,
    dealias: bool = True,
) -> np.ndarray:
    """
    Pseudo-spectral solver for the 2D scalar viscous Burgers equation:
        u_t + u u_x + u u_y = nu (u_xx + u_yy)
    with periodic boundary conditions on [0, 2pi)^2.

    Time integration: integrating-factor Euler
        û^{n+1} = exp(-nu k^2 dt) * (û^n + dt * F̂[nonlinear]^n)
    where k^2 = kx^2 + ky^2.  The linear diffusion is treated exactly via
    the exponential integrating factor, which removes the diffusive CFL
    restriction.  The remaining (hyperbolic) CFL constraint is
        dt * |u|_max / dx  <  1.
    Aliasing errors in the nonlinear term are suppressed with the 2/3 rule
    when `dealias=True`.

    Args:
        u0:      Initial condition, shape (H, W), values on [0, 2pi)^2.
        nu:      Kinematic viscosity.
        dt:      Time step.
        T_steps: Number of time steps to record *after* the IC
                 (total output frames = T_steps + 1, including t=0).
        dealias: Apply 2/3 de-aliasing to the nonlinear term.

    Returns:
        solutions: float32 array of shape (T_steps+1, H, W).
    """
    H, W = u0.shape
    KX, KY = _wavenumbers(H, W)
    K2 = KX ** 2 + KY ** 2                # (H, W//2+1)
    exp_factor = np.exp(-nu * K2 * dt)     # exact diffusion integrating factor

    # De-aliasing mask (2/3 rule in each direction)
    if dealias:
        kx_max = W // 3
        ky_max = H // 3
        dealias_mask = (np.abs(KX) < kx_max) & (np.abs(KY) < ky_max)
    else:
        dealias_mask = np.ones_like(K2, dtype=bool)

    u = u0.copy()
    solutions = [u.astype(np.float32)]

    for _ in range(T_steps):
        u_hat = np.fft.rfft2(u)

        if dealias:
            u_hat_da = u_hat * dealias_mask
        else:
            u_hat_da = u_hat

        # Spectral derivatives of de-aliased field
        ux = np.fft.irfft2(1j * KX * u_hat_da, s=(H, W))
        uy = np.fft.irfft2(1j * KY * u_hat_da, s=(H, W))

        # Nonlinear term in physical space, back to spectral
        nonlin_hat = np.fft.rfft2(-u * ux - u * uy)
        if dealias:
            nonlin_hat *= dealias_mask

        # Integrating-factor Euler step
        u_hat_new = (u_hat + dt * nonlin_hat) * exp_factor
        u = np.fft.irfft2(u_hat_new, s=(H, W))
        solutions.append(u.astype(np.float32))

    return np.stack(solutions, axis=0)  # (T_steps+1, H, W)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    out_dir: str,
    nu_list: tuple,
    N_per_nu: int,
    H: int,
    W: int,
    T_steps: int,
    dt: float,
    n_modes: int,
    global_seed: int,
    dealias: bool,
) -> None:
    """Generate one HDF5 file per viscosity value and save to `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)

    x_coord = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False).astype(np.float32)
    y_coord = np.linspace(0.0, 2.0 * np.pi, H, endpoint=False).astype(np.float32)
    t_coord = (np.arange(T_steps + 1) * dt).astype(np.float32)

    for nu in nu_list:
        out_path = os.path.join(out_dir, f"2D_Burgers_Sols_Nu{nu}.hdf5")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists — delete to regenerate")
            continue

        print(f"Generating nu={nu}: {N_per_nu} samples, grid={H}x{W}, T={T_steps+1} ...")
        tensor = np.empty((N_per_nu, T_steps + 1, H, W), dtype=np.float32)

        for n in range(N_per_nu):
            seed = global_seed + int(nu * 1e6) + n
            rng = np.random.default_rng(seed)
            u0 = _random_ic(H, W, n_modes=n_modes, rng=rng)
            traj = solve_burgers2d(u0, nu=nu, dt=dt, T_steps=T_steps, dealias=dealias)
            tensor[n] = traj  # (T+1, H, W)

            if (n + 1) % 10 == 0:
                print(f"  nu={nu}  sample {n+1}/{N_per_nu}")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("tensor",       data=tensor,   compression="gzip", chunks=True)
            f.create_dataset("t-coordinate", data=t_coord)
            f.create_dataset("x-coordinate", data=x_coord)
            f.create_dataset("y-coordinate", data=y_coord)

        print(f"  Saved {out_path}  shape={tensor.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D viscous Burgers equation HDF5 datasets."
    )
    default_out = "/pscratch/sd/g/gzhao27/INR/data"
    parser.add_argument("--out_dir",    default=default_out,  help="Output directory")
    parser.add_argument("--H",          type=int,   default=512,    help="Grid height")
    parser.add_argument("--W",          type=int,   default=512,    help="Grid width")
    parser.add_argument("--T_steps",    type=int,   default=100,   help="Number of output time steps (total frames = T_steps+1)")
    parser.add_argument("--dt",         type=float, default=0.01, help="Time step size")
    parser.add_argument("--N_train_nu", type=int,   default=20,   help="Samples per nu (train viscosities)")
    parser.add_argument("--N_val_nu",   type=int,   default=20,   help="Samples per nu (val viscosities)")
    parser.add_argument("--n_modes",    type=int,   default=8,     help="Number of IC Fourier modes per axis")
    parser.add_argument("--seed",       type=int,   default=42,    help="Global base random seed")
    parser.add_argument("--no_dealias", action="store_true",       help="Disable 2/3 de-aliasing")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Train viscosities (5 values) + validation (1) + test (1)
    train_nu = (0.001, 0.002, 0.005, 0.02, 0.05)
    val_nu   = (0.01,)
    test_nu  = (0.002,)   # overlaps train intentionally (different sample indices)

    all_nu_to_N = {nu: args.N_train_nu for nu in train_nu}
    for nu in val_nu:
        all_nu_to_N[nu] = all_nu_to_N.get(nu, 0) + args.N_val_nu
    for nu in test_nu:
        all_nu_to_N[nu] = all_nu_to_N.get(nu, 0) + args.N_val_nu  # test uses same N

    # Flatten to a single list; for shared nu (e.g. nu=0.002 in both train & test)
    # we generate the larger count so all index ranges are covered.
    unique_nu = sorted(set(list(train_nu) + list(val_nu) + list(test_nu)))

    generate_dataset(
        out_dir=args.out_dir,
        nu_list=unique_nu,
        N_per_nu=max(args.N_train_nu, args.N_val_nu),
        H=args.H,
        W=args.W,
        T_steps=args.T_steps,
        dt=args.dt,
        n_modes=args.n_modes,
        global_seed=args.seed,
        dealias=not args.no_dealias,
    )
    print("Done.")
