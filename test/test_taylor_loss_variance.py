import argparse
import math

import torch
import torch.nn as nn
from torch_geometric.data import Data
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../"))

from train_utility_sampling.taylor_estimation import (
    cell_loss_variance_estimate_with_random_sampling,
    cell_loss_variance_estimate_with_taylor,
    loss_variance_ground_truth,
)


class AnalyticINR(nn.Module):
    """Simple analytic INR-like model used for deterministic testing."""

    def __init__(self, a: float, b: float, c: float, d: float = 0.0):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[..., 0]
        y = coords[..., 1]
        out = (
            self.a * torch.sin(3.0 * math.pi * x)
            + self.b * torch.cos(2.0 * math.pi * y)
            + self.c * x * y
            + self.d
        )
        return out.unsqueeze(-1)


def build_synthetic_graph(H: int = 64) -> Data:
    """Create a square graph with fields expected by taylor_estimation helpers."""
    xs = torch.linspace(-1.0, 1.0, H)
    ys = torch.linspace(-1.0, 1.0, H)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # Same coordinate convention used in the training code: space_emb is [x, y].
    space_emb = torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()

    rr, cc = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
    cor = torch.stack([rr, cc], dim=-1).reshape(-1, 2).long()

    gt_model = AnalyticINR(a=1.0, b=0.5, c=0.25, d=0.0)
    feat = gt_model(space_emb).detach().float()

    graph = Data(
        cor=cor,
        space_emb=space_emb,
        feat=feat,
        time=torch.zeros(H * H),
        T=torch.tensor(1),
    )
    return graph


def sample_cell_cor_ranges(
    H: int,
    n_cells: int,
    seed: int = 0,
    min_cell_size: int = 3,
    max_cell_size: int = 17,
) -> torch.Tensor:
    """Sample interior square cells as [r_start, r_end, c_start, c_end] (inclusive)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    if min_cell_size < 2:
        raise ValueError("min_cell_size must be >= 2")
    if max_cell_size < min_cell_size:
        raise ValueError("max_cell_size must be >= min_cell_size")

    # Need one-pixel margin so finite-difference neighbors around center stay valid.
    max_allowed_size = H - 2
    if min_cell_size > max_allowed_size:
        raise ValueError(
            f"min_cell_size={min_cell_size} is too large for H={H}; max allowed is {max_allowed_size}"
        )
    max_cell_size = min(max_cell_size, max_allowed_size)

    sizes = torch.randint(min_cell_size, max_cell_size + 1, (n_cells,), generator=g)

    r0_list = []
    c0_list = []
    for size in sizes.tolist():
        # r0/c0 are chosen so r1/c1 are <= H-2, preserving center-neighbor stencil validity.
        r0 = torch.randint(1, H - size, (1,), generator=g).item()
        c0 = torch.randint(1, H - size, (1,), generator=g).item()
        r0_list.append(r0)
        c0_list.append(c0)

    r0 = torch.tensor(r0_list, dtype=torch.long)
    c0 = torch.tensor(c0_list, dtype=torch.long)
    r1 = r0 + sizes - 1
    c1 = c0 + sizes - 1
    return torch.stack([r0, r1, c0, c1], dim=1).long()


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().cpu()
    y = y.float().cpu()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() == 0:
        return 0.0
    return float((x * y).sum() / denom)


def test_zero_loss_case(graph: Data, cell_ranges: torch.Tensor, device: torch.device) -> None:
    """If reconstruction is exact, both estimated and ground-truth loss variance should be ~0."""
    inr_perfect = AnalyticINR(a=1.0, b=0.5, c=0.25, d=0.0).to(device)

    est = cell_loss_variance_estimate_with_taylor(cell_ranges, graph, inr_perfect, device)
    gt = loss_variance_ground_truth(cell_ranges, graph, inr_perfect, device)

    assert est.shape == gt.shape == (cell_ranges.shape[0],), "Unexpected output shape"
    assert torch.isfinite(est).all(), "Estimator output has non-finite values"
    assert torch.isfinite(gt).all(), "Ground-truth output has non-finite values"

    assert torch.max(torch.abs(est)).item() < 1e-8, "Estimator should be ~0 in zero-loss case"
    assert torch.max(torch.abs(gt)).item() < 1e-8, "Ground truth should be ~0 in zero-loss case"


def test_nontrivial_case(
    graph: Data,
    cell_ranges: torch.Tensor,
    device: torch.device,
    min_corr: float,
) -> float:
    """Check estimator basic validity and positive trend against ground truth."""
    inr_imperfect = AnalyticINR(a=0.85, b=0.35, c=0.12, d=0.05).to(device)

    est = cell_loss_variance_estimate_with_taylor(cell_ranges, graph, inr_imperfect, device)
    gt = loss_variance_ground_truth(cell_ranges, graph, inr_imperfect, device)

    assert est.shape == gt.shape == (cell_ranges.shape[0],), "Unexpected output shape"
    assert torch.isfinite(est).all(), "Estimator output has non-finite values"
    assert (est >= 0).all(), "Estimator output should be non-negative"

    corr = pearson_corr(est, gt)
    assert corr >= min_corr, f"Correlation too low: corr={corr:.4f}, min_corr={min_corr:.4f}"
    return corr


def test_random_sampling_case(
    graph: Data,
    cell_ranges: torch.Tensor,
    device: torch.device,
    min_corr_random: float,
    seed: int,
    repeats: int,
) -> float:
    """Check random-sampling estimator validity and trend against ground truth."""
    inr_imperfect = AnalyticINR(a=0.85, b=0.35, c=0.12, d=0.05).to(device)
    gt = loss_variance_ground_truth(cell_ranges, graph, inr_imperfect, device)

    est_runs = []
    for k in range(repeats):
        torch.manual_seed(seed + k)
        est_k = cell_loss_variance_estimate_with_random_sampling(
            cell_ranges, graph, inr_imperfect, device
        )
        est_runs.append(est_k)
    est = torch.stack(est_runs, dim=0).mean(dim=0)

    assert est.shape == gt.shape == (cell_ranges.shape[0],), "Unexpected output shape"
    assert torch.isfinite(est).all(), "Random estimator output has non-finite values"
    assert (est >= 0).all(), "Random estimator output should be non-negative"

    corr = pearson_corr(est, gt)
    assert corr >= min_corr_random, (
        f"Random estimator correlation too low: corr={corr:.4f}, min_corr_random={min_corr_random:.4f}"
    )
    return corr


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Taylor loss-variance estimator on synthetic graph")
    parser.add_argument("--H", type=int, default=64, help="Image/grid resolution")
    parser.add_argument("--n-cells", type=int, default=64, help="Number of sampled test cells")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for cell sampling")
    parser.add_argument("--min-cell-size", type=int, default=3, help="Minimum sampled cell size (inclusive)")
    parser.add_argument("--max-cell-size", type=int, default=17, help="Maximum sampled cell size (inclusive)")
    parser.add_argument("--min-corr", type=float, default=0.1, help="Minimum Pearson correlation threshold")
    parser.add_argument("--min-corr-random", type=float, default=0.05, help="Minimum correlation threshold for random-sampling estimator")
    parser.add_argument("--random-repeats", type=int, default=4, help="Number of random-estimator runs to average")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    graph = build_synthetic_graph(H=args.H)
    cell_ranges = sample_cell_cor_ranges(
        H=args.H,
        n_cells=args.n_cells,
        seed=args.seed,
        min_cell_size=args.min_cell_size,
        max_cell_size=args.max_cell_size,
    )

    test_zero_loss_case(graph=graph, cell_ranges=cell_ranges, device=device)
    corr = test_nontrivial_case(
        graph=graph,
        cell_ranges=cell_ranges,
        device=device,
        min_corr=args.min_corr,
    )
    corr_random = test_random_sampling_case(
        graph=graph,
        cell_ranges=cell_ranges,
        device=device,
        min_corr_random=args.min_corr_random,
        seed=args.seed,
        repeats=args.random_repeats,
    )

    print("[PASS] Loss-variance estimator checks completed.")
    print(
        f"  H={args.H}, n_cells={args.n_cells}, seed={args.seed}, "
        f"cell_size=[{args.min_cell_size}, {args.max_cell_size}], "
        f"corr_taylor={corr:.4f}, corr_random={corr_random:.4f}"
    )


if __name__ == "__main__":
    main()
