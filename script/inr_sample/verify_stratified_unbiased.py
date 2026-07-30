"""Numerically verify that fixed-grid stratified sampling is unbiased.

The `2d_grid_stratified` / `3d_grid_stratified` samplers weight each drawn point by
``N_h / n~_h`` (Hansen-Hurwitz, using the *expected* allocation). The claim that makes
this a legitimate "unbiased" baseline is

    E[ (1/N) * sum_h sum_{j<=n_h} (N_h / n~_h) * L_hj ]  ==  (1/N) * sum_i L_i

i.e. the sampled weighted loss is an unbiased estimate of the full-grid mean loss.
This script checks that empirically, and contrasts it with the tempting-but-biased
variant that divides by the *realised* count n_h instead (cells drawing zero samples
then drop out, systematically under-counting low-variance regions).

The INR is a fixed network in eval mode, so every point's loss L_i is a deterministic
constant and the only randomness is the sampler itself.

Run:  python script/inr_sample/verify_stratified_unbiased.py
"""

import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train_utility_sampling.SamplerWrapper import (  # noqa: E402
    INRSingle2dSamplerWrapper,
    INRSingle3dSamplerWrapper,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
REPS = int(os.environ.get("VERIFY_REPS", 4000))


class TinyINR(torch.nn.Module):
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
    # Strongly inhomogeneous field: the left half is nearly flat, the right half is
    # rough. Neyman must therefore allocate very unevenly -- exactly the regime where
    # a wrong weighting shows up as bias.
    rough = (space_emb[:, 1] > 0).float()
    field = torch.sin(3 * space_emb[:, 0]) + rough * 2.0 * torch.sin(25 * space_emb[:, 1])
    feat = (field + 0.2 * torch.randn(H * H, generator=g)).unsqueeze(1)
    return Data(cor=cor, space_emb=space_emb, feat=feat,
                time=torch.zeros(H * H, dtype=torch.long))


def make_3d_graph(D=(24, 24, 8), hole_frac=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    Dx, Dy, Dz = D
    xx, yy, zz = torch.meshgrid(
        torch.arange(Dx), torch.arange(Dy), torch.arange(Dz), indexing="ij"
    )
    cor = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
    keep = torch.rand(cor.size(0), generator=g) > hole_frac  # emulate a land/ocean mask
    cor = cor[keep]
    space_emb = cor.float() / torch.tensor(D, dtype=torch.float32) * 2 - 1
    rough = (space_emb[:, 0] > 0).float()
    field = torch.sin(3 * space_emb[:, 1]) + rough * 2.0 * torch.sin(20 * space_emb[:, 0])
    feat = (field + 0.2 * torch.randn(cor.size(0), generator=g)).unsqueeze(1)
    return Data(cor=cor, space_emb=space_emb, feat=feat,
                time=torch.zeros(cor.size(0), dtype=torch.long))


@torch.no_grad()
def true_mean_loss(graph, inr):
    preds = inr(graph.space_emb.to(DEVICE))
    return (preds - graph.feat.to(DEVICE)).pow(2).sum(dim=1).mean().item()


@torch.no_grad()
def run_trials(sampler, graph, inr, n_total, reps):
    """Return (horvitz_thompson_estimates, self_normalised_estimates) over `reps` draws."""
    ht, sn = [], []
    for step in range(reps):
        # step beyond the refresh interval on the first call only, so sigma_h is
        # estimated once and then held fixed -- matching a stretch of training.
        sub = sampler.sample(step, graph.clone(), save_image=False)
        preds = inr(sub.space_emb.to(DEVICE))
        losses = (preds - sub.feat.to(DEVICE)).pow(2).sum(dim=1)
        w = sub.weight.to(DEVICE).reshape(-1)
        ht.append((w * losses).sum().item() / n_total)
        sn.append(((w * losses).sum() / w.sum()).item())
    return torch.tensor(ht), torch.tensor(sn)


def report(name, est, truth, tol_z=4.0):
    mean, std = est.mean().item(), est.std().item()
    se = std / (est.numel() ** 0.5)
    z = abs(mean - truth) / max(se, 1e-15)
    rel = (mean - truth) / truth
    ok = z < tol_z
    print(f"  {name:<34s} mean={mean:.6e}  bias={rel:+.4%}  z={z:6.2f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_2d():
    print("\n=== 2D  (64x64 grid, 8x8 = 64 cells, n = 256 samples) ===")
    torch.manual_seed(0)
    graph = make_2d_graph()
    inr = TinyINR(2).to(DEVICE).eval()
    n_points = graph.cor.size(0)
    rate = 256 / n_points
    truth = true_mean_loss(graph, inr)
    print(f"  true full-grid mean loss = {truth:.6e}")

    results = []
    for allocation in ("proportional", "neyman"):
        sampler = INRSingle2dSamplerWrapper(
            model=inr, iters=0, device=DEVICE, sample_type="2d_grid_stratified",
            sample_rate=rate, image_width=64,
            stratified_allocation=allocation, stratified_n_bins=8,
            stratified_min_alloc_frac=0.1, stratified_update_interval=10 ** 9,
        )
        ht, sn = run_trials(sampler, graph, inr, n_points, REPS)
        print(f"  --- allocation={allocation} ---")
        results.append(report("Horvitz-Thompson (unbiased)", ht, truth))
        # Self-normalised is a ratio estimator: consistent, with O(1/n) bias. Reported
        # for transparency, not asserted -- it is what the training loss actually uses.
        report("self-normalised (training loss)", sn, truth, tol_z=float("inf"))
    return results


def check_2d_realised_count_is_biased():
    """Control: the N_h / n_h variant should visibly fail the same test."""
    print("\n=== 2D control: weighting by the REALISED count n_h (expected to be biased) ===")
    torch.manual_seed(0)
    graph = make_2d_graph()
    inr = TinyINR(2).to(DEVICE).eval()
    n_points = graph.cor.size(0)
    truth = true_mean_loss(graph, inr)

    from train_utility_sampling.SamplerWrapper import (
        build_uniform_grid_bounds_2d,
        sample_variable_from_2d_intervals_vcounts,
        stratified_continuous_allocation,
        stratified_poisson_counts,
    )

    bounds = build_uniform_grid_bounds_2d(64, 8, device="cpu")
    cell_n = ((bounds[:, 1] - bounds[:, 0] + 1) * (bounds[:, 3] - bounds[:, 2] + 1)).float().to(DEVICE)
    n_total = 256
    with torch.no_grad():
        preds_all = inr(graph.space_emb.to(DEVICE))
        loss_all = (preds_all - graph.feat.to(DEVICE)).pow(2).sum(dim=1)

    est = []
    for _ in range(REPS):
        alloc = stratified_continuous_allocation(cell_n, n_total, 0.1)
        counts = stratified_poisson_counts(alloc)
        xy, cell_ids, _ = sample_variable_from_2d_intervals_vcounts(bounds, counts, device=DEVICE)
        if xy.numel() == 0:
            continue
        idx = xy[:, 1] * 64 + xy[:, 0]
        realised = counts.to(DEVICE).clamp_min(1).float()
        w = (cell_n / realised)[cell_ids]
        est.append((w * loss_all[idx]).sum().item() / n_points)
    est = torch.tensor(est)
    biased = not report("N_h / n_h  (realised count)", est, truth)
    print(f"  -> {'confirmed biased' if biased else 'NOT detectably biased at this REPS'}")
    return True  # informational control, never gates the result


def check_3d():
    print("\n=== 3D  (24x24x8 volume, 20% holes, 3x3x2 = 18 cells, n = 256 samples) ===")
    torch.manual_seed(0)
    D = (24, 24, 8)
    graph = make_3d_graph(D)
    inr = TinyINR(3).to(DEVICE).eval()
    n_points = graph.cor.size(0)
    rate = 256 / n_points
    truth = true_mean_loss(graph, inr)
    print(f"  true full-grid mean loss = {truth:.6e}  ({n_points} valid voxels)")

    results = []
    for allocation in ("proportional", "neyman"):
        sampler = INRSingle3dSamplerWrapper(
            model=inr, iters=0, device=DEVICE, sample_type="3d_grid_stratified",
            sample_rate=rate, grid_shape=D,
            stratified_allocation=allocation, stratified_n_bins=[3, 3, 2],
            stratified_min_alloc_frac=0.1, stratified_update_interval=10 ** 9,
        )
        ht, sn = run_trials(sampler, graph, inr, n_points, REPS)
        print(f"  --- allocation={allocation} ---")
        results.append(report("Horvitz-Thompson (unbiased)", ht, truth))
        report("self-normalised (training loss)", sn, truth, tol_z=float("inf"))
    return results


def main():
    results = []
    results += check_2d()
    check_2d_realised_count_is_biased()
    results += check_3d()

    print("\n" + "=" * 70)
    if all(results):
        print("ALL UNBIASEDNESS CHECKS PASSED.")
        return 0
    print("SOME CHECKS FAILED - the stratified estimator is NOT unbiased as implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
