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
from typing import Optional, Sequence

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


def _random_ic(
    H: int,
    W: int,
    n_modes: int = 8,
    rng: Optional[np.random.Generator] = None,
    decay_power: float = 2.0,
    highfreq_boost: float = 0.0,
    target_rms: float = 1.0,
) -> np.ndarray:
    """
        Random initial condition: superposition of Fourier modes on [0, 2pi)^2.

        Amplitudes follow:
            amp ~ 1 / (k^2 + l^2)^(decay_power / 2)
        with an optional multiplicative high-frequency boost term:
            ((sqrt(k^2+l^2) / sqrt(2)) ** highfreq_boost)

        Lower `decay_power` and/or higher `highfreq_boost` produce more complex
        fields by allocating more energy to higher frequencies.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, H, endpoint=False)
    X, Y = np.meshgrid(x, y)
    u = np.zeros((H, W), dtype=np.float64)
    for k in range(1, n_modes + 1):
        for l in range(1, n_modes + 1):
            k2 = float(k ** 2 + l ** 2)
            radial = np.sqrt(k2)
            # Base spectral decay (smoothness control).
            amp_scale = 1.0 / (k2 ** (0.5 * decay_power))
            # Optional boost for higher radial frequencies.
            if highfreq_boost != 0.0:
                amp_scale *= (radial / np.sqrt(2.0)) ** highfreq_boost

            amp = rng.uniform(-1.0, 1.0) * amp_scale
            phase = rng.uniform(0.0, 2.0 * np.pi)
            u += amp * np.sin(k * X + l * Y + phase)

    # Keep a predictable amplitude scale while preserving spectral shape.
    rms = np.sqrt(np.mean(u ** 2))
    if rms > 0.0:
        u *= target_rms / rms
    return u


def solve_burgers2d(
    u0: np.ndarray,
    nu: float,
    dt: float,
    T_steps: int,
    dealias: bool = True,
    substeps: int = 1,
) -> np.ndarray:
    """
    Pseudo-spectral solver for the 2D scalar viscous Burgers equation:
        u_t + u u_x + u u_y = nu (u_xx + u_yy)
    with periodic boundary conditions on [0, 2pi)^2.

    Time integration: classical RK4 applied to the full spectral RHS
        RHS(û) = -nu k^2 û + N̂(û)
    where k^2 = kx^2 + ky^2 and N̂ is the (dealiased) nonlinear advection.
    RK4 has a 4x larger stability region for the advection term compared to
    forward Euler, making it robust once Burgers gradients steepen.

    The Courant constraint is:
        dt * |u|_max / dx  <  ~2.8   (RK4 vs ~1.0 for Euler)

    Args:
        u0:       Initial condition, shape (H, W), values on [0, 2pi)^2.
        nu:       Kinematic viscosity.
        dt:       Internal time step (controls stability).
        T_steps:  Number of output frames to record *after* the IC
                  (total output frames = T_steps + 1, including t=0).
        dealias:  Apply 2/3 de-aliasing to the nonlinear term.
        substeps: Number of internal `dt` steps taken per recorded output
                  frame.  Physical time between frames = dt * substeps.
                  Increase this to make the solution evolve more between
                  saved frames without changing stability (dt stays small).

    Returns:
        solutions: float32 array of shape (T_steps+1, H, W).
    """
    H, W = u0.shape
    KX, KY = _wavenumbers(H, W)
    K2 = KX ** 2 + KY ** 2  # (H, W//2+1)

    # De-aliasing mask (2/3 rule in each direction)
    if dealias:
        kx_max = W // 3
        ky_max = H // 3
        dealias_mask = (np.abs(KX) < kx_max) & (np.abs(KY) < ky_max)
    else:
        dealias_mask = np.ones_like(K2, dtype=bool)

    def rhs(u_hat: np.ndarray) -> np.ndarray:
        """Full spectral RHS: diffusion + dealias-filtered nonlinear advection."""
        u_hat_da = u_hat * dealias_mask if dealias else u_hat
        u_phys = np.fft.irfft2(u_hat_da, s=(H, W))
        ux = np.fft.irfft2(1j * KX * u_hat_da, s=(H, W))
        uy = np.fft.irfft2(1j * KY * u_hat_da, s=(H, W))
        nl_hat = np.fft.rfft2(-u_phys * ux - u_phys * uy)
        if dealias:
            nl_hat *= dealias_mask
        return -nu * K2 * u_hat + nl_hat

    u_hat = np.fft.rfft2(u0.copy())
    solutions = [np.fft.irfft2(u_hat, s=(H, W)).astype(np.float32)]

    for _ in range(T_steps):
        # Take `substeps` internal RK4 steps before recording this output frame.
        for _ in range(substeps):
            k1 = rhs(u_hat)
            k2 = rhs(u_hat + 0.5 * dt * k1)
            k3 = rhs(u_hat + 0.5 * dt * k2)
            k4 = rhs(u_hat + dt * k3)
            u_hat = u_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        solutions.append(np.fft.irfft2(u_hat, s=(H, W)).astype(np.float32))

    return np.stack(solutions, axis=0)  # (T_steps+1, H, W)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    out_dir: str,
    nu_list: Sequence[float],
    N_per_nu: int,
    H: int,
    W: int,
    T_steps: int,
    dt: float,
    substeps: int,
    n_modes: int,
    ic_decay_power: float,
    ic_highfreq_boost: float,
    ic_target_rms: float,
    global_seed: int,
    dealias: bool,
) -> None:
    """Generate one HDF5 file per viscosity value and save to `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)

    x_coord = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False).astype(np.float32)
    y_coord = np.linspace(0.0, 2.0 * np.pi, H, endpoint=False).astype(np.float32)
    # t_coord reflects physical time: each frame is dt * substeps apart.
    t_coord = (np.arange(T_steps + 1) * dt * substeps).astype(np.float32)

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
            u0 = _random_ic(
                H,
                W,
                n_modes=n_modes,
                rng=rng,
                decay_power=ic_decay_power,
                highfreq_boost=ic_highfreq_boost,
                target_rms=ic_target_rms,
            )
            traj = solve_burgers2d(u0, nu=nu, dt=dt, T_steps=T_steps, dealias=dealias, substeps=substeps)
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
    default_out = "/pscratch/sd/g/gzhao27/INR/data/2D_Burgers_Sols"
    parser.add_argument("--out_dir",    default=default_out,  help="Output directory")
    parser.add_argument("--H",          type=int,   default=1024,    help="Grid height")
    parser.add_argument("--W",          type=int,   default=1024,    help="Grid width")
    parser.add_argument("--T_steps",    type=int,   default=100,   help="Number of output time steps (total frames = T_steps+1)")
    parser.add_argument("--dt",         type=float, default=0.001, help="Internal time step size (controls stability)")
    parser.add_argument(
        "--substeps",
        type=int,
        default=50,
        help="Internal solver steps per recorded output frame; physical time per frame = dt * substeps",
    )
    parser.add_argument("--N_train_nu", type=int,   default=10,   help="Samples per nu (train viscosities)")
    parser.add_argument("--n_modes",    type=int,   default=8,     help="Number of IC Fourier modes per axis")
    parser.add_argument(
        "--ic_decay_power",
        type=float,
        default=2.0,
        help="Spectral decay power for IC amplitudes; smaller -> richer high-freq content",
    )
    parser.add_argument(
        "--ic_highfreq_boost",
        type=float,
        default=0.0,
        help="Additional high-frequency emphasis for IC amplitudes",
    )
    parser.add_argument(
        "--ic_target_rms",
        type=float,
        default=1.0,
        help="RMS normalization target for initial condition amplitude",
    )
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

    # Flatten to a single list; for shared nu (e.g. nu=0.002 in both train & test)
    # we generate the larger count so all index ranges are covered.
    unique_nu = sorted(set(list(train_nu) + list(val_nu) + list(test_nu)))
    generate_dataset(
        out_dir=args.out_dir,
        nu_list=unique_nu,
        N_per_nu=args.N_train_nu,
        H=args.H,
        W=args.W,
        T_steps=args.T_steps,
        dt=args.dt,
        substeps=args.substeps,
        n_modes=args.n_modes,
        ic_decay_power=args.ic_decay_power,
        ic_highfreq_boost=args.ic_highfreq_boost,
        ic_target_rms=args.ic_target_rms,
        global_seed=args.seed,
        dealias=not args.no_dealias,
    )
    print("Done.")
