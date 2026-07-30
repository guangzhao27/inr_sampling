"""Verify that the batched per-cell pilot variance estimators are a pure refactor.

The batched implementations (`cell_loss_variance_batched`, `cell_loss_variance_batched_3d`)
replace per-cell Python loops with a single forward pass. They cannot reproduce the loop
versions bit-for-bit -- the RNG stream differs -- so equivalence is checked *in distribution*:

  1. Per-cell Monte-Carlo means of the returned variance must agree within MC error
     (paired z-score over R independent repetitions).
  2. The pooled distribution of returned values must pass a two-sample KS test.
  3. Degenerate cells (single voxel, all-hole cells in 3D) must return exactly 0 in both.

A real bias grows as sqrt(REPS) while MC noise does not, so re-run with a larger
`VERIFY_REPS` to tell the two apart:  VERIFY_REPS=1200 python ...

The pre-refactor loop implementations are kept verbatim in this file as
`reference_*_loop`, so this stays a genuine equivalence test after the production
estimators in taylor_estimation.py were switched over to delegate to the batched core.

Run:  python script/inr_sample/verify_batched_estimator.py
"""

import os
import sys
from pathlib import Path

import torch
from scipy.stats import ks_2samp
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train_utility_sampling.taylor_estimation import (  # noqa: E402
    build_3d_index_grid,
    cell_loss_variance_batched,
    cell_loss_variance_batched_3d,
    loss_function,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
REPS = int(os.environ.get("VERIFY_REPS", 300))


# ---------------------------------------------------------------------------
# Reference implementations: the original per-cell loops, kept verbatim.
# ---------------------------------------------------------------------------

def reference_2d_loop(cell_cor_range, graph, inr, device, max_samples_per_cell=16, use_sqrt=False):
    graph = graph.cpu()
    H = graph.cor.max().item() + 1
    features = graph.feat.view(H, H, 1)
    coords = graph.space_emb.view(H, H, 2)
    inr.to(device)

    var_list = []
    for cell in cell_cor_range:
        r_start, r_end, c_start, c_end = [int(v) for v in cell.tolist()]
        h = r_end - r_start + 1
        w = c_end - c_start + 1
        n_samples = min(max_samples_per_cell, h * w)

        rr = torch.randint(r_start, r_end + 1, (n_samples,))
        cc = torch.randint(c_start, c_end + 1, (n_samples,))

        with torch.no_grad():
            sample_recon = inr(coords[rr, cc].to(device))
        losses = loss_function(sample_recon, features[rr, cc].to(device)).reshape(-1)
        if use_sqrt:
            losses = losses.sqrt()
        var_list.append(losses.var(unbiased=False))
    return torch.stack(var_list, dim=0)


def reference_3d_loop(cell_cor_range, grid_shape, graph, inr, device,
                      max_samples_per_cell=16, use_sqrt=False):
    graph = graph.cpu()
    index_grid = build_3d_index_grid(graph, grid_shape)
    inr.to(device)

    var_list = []
    for cell in cell_cor_range:
        x_start, x_end, y_start, y_end, z_start, z_end = [int(v) for v in cell.tolist()]
        w = x_end - x_start + 1
        h = y_end - y_start + 1
        d = z_end - z_start + 1
        n_candidates = min(max_samples_per_cell, w * h * d)

        xx = torch.randint(x_start, x_end + 1, (n_candidates,))
        yy = torch.randint(y_start, y_end + 1, (n_candidates,))
        zz = torch.randint(z_start, z_end + 1, (n_candidates,))

        node_idx = index_grid[xx, yy, zz]
        node_idx = node_idx[node_idx >= 0]
        if node_idx.numel() == 0:
            var_list.append(torch.zeros((), device=device))
            continue

        with torch.no_grad():
            sample_recon = inr(graph.space_emb[node_idx].to(device))
        losses = loss_function(sample_recon, graph.feat[node_idx].to(device)).reshape(-1)
        if use_sqrt:
            losses = losses.sqrt()
        var_list.append(losses.var(unbiased=False))
    return torch.stack(var_list, dim=0)


class TinyINR(torch.nn.Module):
    """Small fixed MLP standing in for the real SIREN; only needs to be a smooth map."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_2d_graph(H=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
    cor = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    space_emb = (cor.float() / (H - 1)) * 2 - 1
    # Structured field + noise so per-cell variances differ a lot across the image.
    field = torch.sin(6 * space_emb[:, 0]) * torch.cos(4 * space_emb[:, 1])
    feat = (field + 0.3 * torch.randn(H * H, generator=g)).unsqueeze(1)
    return Data(cor=cor, space_emb=space_emb, feat=feat)


def make_2d_cells(H=64):
    """Mixed-size cells, including 1x1 and sub-16-pixel cells to exercise the min() path."""
    cells = []
    for r in range(0, H, 16):                       # 16 large 16x16 cells
        for c in range(0, H, 16):
            cells.append([r, r + 15, c, c + 15])
    for r in range(0, 12, 3):                       # small 3x3 cells (9 < 16 pilot points)
        cells.append([r, r + 2, 0, 2])
    cells.append([5, 5, 7, 7])                      # degenerate 1x1 cell
    cells.append([9, 9, 3, 3])
    return torch.tensor(cells, dtype=torch.long)


def make_3d_graph(D=(24, 24, 8), hole_frac=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    Dx, Dy, Dz = D
    xx, yy, zz = torch.meshgrid(
        torch.arange(Dx), torch.arange(Dy), torch.arange(Dz), indexing="ij"
    )
    cor = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
    keep = torch.rand(cor.size(0), generator=g) > hole_frac   # emulate a land/ocean mask
    cor = cor[keep]
    space_emb = cor.float() / torch.tensor(D, dtype=torch.float32) * 2 - 1
    field = torch.sin(5 * space_emb[:, 0]) * torch.cos(3 * space_emb[:, 1]) * space_emb[:, 2]
    feat = (field + 0.3 * torch.randn(cor.size(0), generator=g)).unsqueeze(1)
    return Data(cor=cor, space_emb=space_emb, feat=feat)


def make_3d_cells(D=(24, 24, 8)):
    Dx, Dy, Dz = D
    cells = []
    for x in range(0, Dx, 8):
        for y in range(0, Dy, 8):
            for z in range(0, Dz, 4):
                cells.append([x, x + 7, y, y + 7, z, z + 3])
    cells.append([0, 0, 0, 0, 0, 0])                # degenerate single voxel
    cells.append([1, 1, 1, 1, 1, 1])
    return torch.tensor(cells, dtype=torch.long)


def repeat(fn, reps):
    return torch.stack([fn().detach().float().cpu() for _ in range(reps)], dim=0)  # (reps, n_cells)


def compare(name, loop_samples, batch_samples, tol_z=4.0, ks_alpha=1e-3):
    """Compare two (reps, n_cells) Monte-Carlo sample stacks."""
    lm, bm = loop_samples.mean(0), batch_samples.mean(0)
    ls, bs = loop_samples.std(0), batch_samples.std(0)
    reps = loop_samples.size(0)

    se = torch.sqrt((ls.pow(2) + bs.pow(2)) / reps).clamp_min(1e-12)
    z = (lm - bm).abs() / se
    # Cells where both estimators are exactly deterministic (e.g. 1x1 cells -> variance 0).
    deterministic = (ls < 1e-12) & (bs < 1e-12)
    z = torch.where(deterministic, torch.zeros_like(z), z)

    exact_mismatch = deterministic & ((lm - bm).abs() > 1e-12)
    ks = ks_2samp(loop_samples.reshape(-1).numpy(), batch_samples.reshape(-1).numpy())

    max_z = z.max().item()
    rel_bias = ((lm - bm).abs().sum() / lm.abs().sum().clamp_min(1e-12)).item()
    ok = (max_z < tol_z) and (ks.pvalue > ks_alpha) and (not exact_mismatch.any())

    print(f"\n--- {name} ---")
    print(f"  cells={loop_samples.size(1)}  reps={reps}")
    print(f"  max paired z-score      : {max_z:.3f}   (threshold {tol_z})")
    print(f"  pooled KS statistic     : {ks.statistic:.5f}  p={ks.pvalue:.4f} (threshold {ks_alpha})")
    print(f"  aggregate relative bias : {rel_bias:.5f}")
    print(f"  deterministic cells     : {int(deterministic.sum())} (mismatches: {int(exact_mismatch.sum())})")
    print(f"  RESULT                  : {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    results = []

    # ---------------- 2D ----------------
    graph2d = make_2d_graph()
    cells2d = make_2d_cells()
    inr2d = TinyINR(2).to(DEVICE).eval()

    for use_sqrt in (False, True):
        loop = repeat(
            lambda: reference_2d_loop(cells2d, graph2d.clone(), inr2d, DEVICE, use_sqrt=use_sqrt),
            REPS,
        )
        batch = repeat(
            lambda: cell_loss_variance_batched(
                cells2d, graph2d.clone(), inr2d, DEVICE, use_sqrt=use_sqrt
            ),
            REPS,
        )
        results.append(compare(f"2D  use_sqrt={use_sqrt}", loop, batch))

    # ---------------- 3D ----------------
    D = (24, 24, 8)
    graph3d = make_3d_graph(D)
    cells3d = make_3d_cells(D)
    inr3d = TinyINR(3).to(DEVICE).eval()
    idx_grid = build_3d_index_grid(graph3d.clone().cpu(), D)

    for use_sqrt in (False, True):
        loop = repeat(
            lambda: reference_3d_loop(cells3d, D, graph3d.clone(), inr3d, DEVICE, use_sqrt=use_sqrt),
            REPS,
        )
        batch = repeat(
            lambda: cell_loss_variance_batched_3d(
                cells3d, D, graph3d.clone(), inr3d, DEVICE,
                use_sqrt=use_sqrt, index_grid=idx_grid,
            ),
            REPS,
        )
        results.append(compare(f"3D  use_sqrt={use_sqrt}", loop, batch))

    print("\n" + "=" * 60)
    if all(results):
        print("ALL CHECKS PASSED - batched estimators are distributionally equivalent.")
        return 0
    print("SOME CHECKS FAILED - do NOT switch the loop estimators over.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
