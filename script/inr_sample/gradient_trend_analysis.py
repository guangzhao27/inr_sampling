#!/usr/bin/env python3
"""Analyze checkpoint gradient stability under repeated sampler draws.

This standalone script scans a parent directory of INR checkpoint run folders,
loads numeric-step checkpoints, repeatedly samples points with a selected
sampler, computes sampled-loss parameter gradients, and reports:

1) Gradient variance across repeated sampling runs
2) Average pairwise cosine correlation across repeated gradient vectors

It writes per-run and per-sampler CSV summaries and trend plots over training
steps.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from train_utility_sampling.SamplerWrapper import create_inr_sampler
from utils.load_inr import create_inr_instance


NUMERIC_CKPT_RE = re.compile(r"^(\d+)\.pt$")


@dataclass
class RepeatMetrics:
    loss: float
    grad: torch.Tensor
    grad_norm: float
    sampled_points: int


@dataclass
class CheckpointMetrics:
    step: int
    grad_variance: float
    avg_pairwise_cosine: float
    orthogonal_grad_error_mean: float
    orthogonal_grad_error_std: float
    loss_mean: float
    loss_std: float
    grad_norm_mean: float
    grad_norm_std: float
    sampled_points_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute checkpoint-wise gradient variance/correlation trends by "
            "repeatedly sampling points with a selected sampler."
        )
    )
    parser.add_argument(
        "--checkpoint-parent",
        type=str,
        required=True,
        help=(
            "Parent directory containing run folders with numeric checkpoint "
            "files (e.g., Results/checkpoints)."
        ),
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="auto",
        help=(
            "Sampler type used during analysis. Use 'auto' to keep each run's "
            "checkpoint sampler, or set to one of {random, NMT, 2d_grid_linear, "
            "2d_grid_adaptive, EVOS, ...}."
        ),
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=4,
        help="Number of repeated sampling/gradient evaluations per checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed; each repeat uses seed + repeat_idx.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Results/gradient_trend_analysis",
        help="Output directory for CSV files, plots, and summary JSON.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Optional limit on number of checkpoints per run (oldest-first).",
    )
    parser.add_argument(
        "--run-name-filter",
        type=str,
        default=None,
        help="Optional substring filter applied to run-folder names.",
    )
    parser.add_argument(
        "--plot-std",
        action="store_true",
        help="If set, draw +/- std shading around sampler-level mean trends.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only first discovered run and first checkpoint.",
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def discover_run_dirs(parent: Path, name_filter: Optional[str]) -> List[Path]:
    run_dirs = [p for p in parent.iterdir() if p.is_dir()]
    if name_filter:
        run_dirs = [p for p in run_dirs if name_filter in p.name]
    return sorted(run_dirs)


def discover_numeric_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    checkpoints: List[Tuple[int, Path]] = []
    for ckpt in run_dir.glob("*.pt"):
        m = NUMERIC_CKPT_RE.match(ckpt.name)
        if m is None:
            continue
        checkpoints.append((int(m.group(1)), ckpt))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def safe_sampler_label(raw_label: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", raw_label)


def build_graph_from_checkpoint(ckpt: Dict, device: torch.device) -> Data:
    graph_data = ckpt.get("graph_data")
    if graph_data is None:
        raise KeyError("checkpoint missing 'graph_data'; cannot reconstruct graph")

    required = ["cor", "space_emb", "feat", "time"]
    missing = [k for k in required if k not in graph_data]
    if missing:
        raise KeyError(f"graph_data missing required keys: {missing}")

    graph_kwargs = {
        "cor": graph_data["cor"],
        "space_emb": graph_data["space_emb"],
        "feat": graph_data["feat"],
        "time": graph_data["time"],
    }

    if "T" in graph_data:
        graph_kwargs["T"] = graph_data["T"]
    if "weight" in graph_data:
        graph_kwargs["weight"] = graph_data["weight"]

    graph = Data(**graph_kwargs)
    return graph.to(device)


def infer_model_dims(ckpt: Dict, graph: Data) -> Tuple[int, int]:
    input_dim = ckpt.get("input_dim")
    output_dim = ckpt.get("output_dim")

    if input_dim is None:
        input_dim = int(graph.space_emb.shape[-1])
    if output_dim is None:
        feat = graph.feat
        output_dim = int(feat.shape[-1] if feat.ndim > 1 else 1)

    return int(input_dim), int(output_dim)


def override_sampler_in_cfg(cfg, sampler_arg: str):
    cfg_eval = copy.deepcopy(cfg)
    if sampler_arg != "auto":
        cfg_eval.sampling.type = sampler_arg
    return cfg_eval


def flatten_model_gradients(model: torch.nn.Module) -> torch.Tensor:
    grads: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return torch.empty(0)
    return torch.cat(grads, dim=0)


def weighted_point_mse(pred: torch.Tensor, target: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    err_sq = (pred - target).pow(2)
    per_point_mse = err_sq.reshape(err_sq.shape[0], -1).mean(dim=1)

    if sample_weight is None:
        return per_point_mse.mean()

    w = sample_weight.to(device=per_point_mse.device, dtype=per_point_mse.dtype).reshape(-1)
    if w.numel() != per_point_mse.numel():
        return per_point_mse.mean()

    w_sum = w.sum()
    if not torch.isfinite(w_sum) or w_sum.item() <= 0:
        return per_point_mse.mean()

    return (w * per_point_mse).sum() / w_sum


def call_sampler(sampler, graph: Data, step: int) -> Data:
    # EVOS uses a different call signature than INRSingle2d samplers.
    if sampler.__class__.__name__ == "EVOSSampler":
        return sampler.sample(graph=graph, epoch=step, inner_step=step, save_image=False)
    return sampler.sample(inner_step=step, graph=graph, save_image=False)


def single_repeat_metrics(
    model: torch.nn.Module,
    base_graph: Data,
    step: int,
    repeat_seed: int,
    sampler,
) -> RepeatMetrics:
    set_all_seeds(repeat_seed)

    model.train()
    model.zero_grad(set_to_none=True)

    sampled_graph = base_graph if sampler is None else call_sampler(sampler, base_graph, step)
    pred = model(sampled_graph.space_emb)
    sample_weight = getattr(sampled_graph, "weight", None)
    loss = weighted_point_mse(pred, sampled_graph.feat, sample_weight)
    loss.backward()

    grad_vec = flatten_model_gradients(model).detach().cpu()
    grad_norm = float(torch.norm(grad_vec).item()) if grad_vec.numel() > 0 else 0.0

    return RepeatMetrics(
        loss=float(loss.detach().item()),
        grad=grad_vec,
        grad_norm=grad_norm,
        sampled_points=int(sampled_graph.feat.shape[0]),
    )


def avg_pairwise_cosine(grads: torch.Tensor) -> float:
    n = grads.shape[0]
    if n < 2:
        return float("nan")

    norms = torch.norm(grads, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = grads / norms
    cosine_mat = normalized @ normalized.t()

    tri = torch.triu_indices(n, n, offset=1)
    vals = cosine_mat[tri[0], tri[1]]
    return float(vals.mean().item())


def full_graph_gradient(
    model: torch.nn.Module,
    graph: Data,
) -> torch.Tensor:
    """Compute gradient using all graph points (no sampler)."""
    model.train()
    model.zero_grad(set_to_none=True)
    pred = model(graph.space_emb)
    loss = weighted_point_mse(pred, graph.feat, sample_weight=None)
    loss.backward()
    return flatten_model_gradients(model).detach().cpu()


def orthogonal_error_stats(sample_grads: torch.Tensor, gt_grad: torch.Tensor) -> Tuple[float, float]:
    """Mean/std of squared orthogonal distance to projection on gt gradient.

    sample_grads: [R, P], gt_grad: [P]
    """
    if sample_grads.numel() == 0 or gt_grad.numel() == 0:
        return float("nan"), float("nan")

    gt = gt_grad.to(sample_grads.dtype)
    gt_norm_sq = torch.dot(gt, gt)

    if not torch.isfinite(gt_norm_sq) or gt_norm_sq.item() <= 1e-20:
        # Degenerate gt direction; fallback to full gradient energy.
        dist_sq = (sample_grads * sample_grads).sum(dim=1)
    else:
        coeff = (sample_grads @ gt) / gt_norm_sq
        proj = coeff.unsqueeze(1) * gt.unsqueeze(0)
        residual = sample_grads - proj
        dist_sq = (residual * residual).sum(dim=1)

    return float(dist_sq.mean().item()), float(dist_sq.std(unbiased=False).item())


def checkpoint_stats_from_repeats(
    repeats: Sequence[RepeatMetrics],
    step: int,
    gt_grad: torch.Tensor,
) -> CheckpointMetrics:
    grads = torch.stack([r.grad for r in repeats], dim=0)
    grad_var = float(grads.var(dim=0, unbiased=False).mean().item())
    pair_cos = avg_pairwise_cosine(grads)
    ortho_mean, ortho_std = orthogonal_error_stats(grads, gt_grad)

    losses = np.asarray([r.loss for r in repeats], dtype=np.float64)
    grad_norms = np.asarray([r.grad_norm for r in repeats], dtype=np.float64)
    sampled_points = np.asarray([r.sampled_points for r in repeats], dtype=np.float64)

    return CheckpointMetrics(
        step=step,
        grad_variance=grad_var,
        avg_pairwise_cosine=pair_cos,
        orthogonal_grad_error_mean=ortho_mean,
        orthogonal_grad_error_std=ortho_std,
        loss_mean=float(losses.mean()),
        loss_std=float(losses.std()),
        grad_norm_mean=float(grad_norms.mean()),
        grad_norm_std=float(grad_norms.std()),
        sampled_points_mean=float(sampled_points.mean()),
    )


def checkpoint_metrics_to_row(m: CheckpointMetrics) -> Dict[str, Any]:
    return {
        "step": m.step,
        "grad_variance": m.grad_variance,
        "avg_pairwise_cosine": m.avg_pairwise_cosine,
        "orthogonal_grad_error_mean": m.orthogonal_grad_error_mean,
        "orthogonal_grad_error_std": m.orthogonal_grad_error_std,
        "loss_mean": m.loss_mean,
        "loss_std": m.loss_std,
        "grad_norm_mean": m.grad_norm_mean,
        "grad_norm_std": m.grad_norm_std,
        "sampled_points_mean": m.sampled_points_mean,
    }


def aggregate_by_step(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("step", as_index=False)
        .agg(
            grad_variance_mean=("grad_variance", "mean"),
            grad_variance_std=("grad_variance", "std"),
            avg_pairwise_cosine_mean=("avg_pairwise_cosine", "mean"),
            avg_pairwise_cosine_std=("avg_pairwise_cosine", "std"),
            orthogonal_grad_error_mean=("orthogonal_grad_error_mean", "mean"),
            orthogonal_grad_error_std=("orthogonal_grad_error_mean", "std"),
            loss_mean=("loss_mean", "mean"),
            grad_norm_mean=("grad_norm_mean", "mean"),
            sampled_points_mean=("sampled_points_mean", "mean"),
            n_runs=("run_name", "nunique"),
        )
        .sort_values("step")
    )
    return grouped


def plot_sampler_metric(
    agg_map: Dict[str, pd.DataFrame],
    metric_key: str,
    ylabel: str,
    output_path: Path,
    plot_std: bool,
) -> None:
    plt.figure(figsize=(8, 5))

    for sampler_label, df in sorted(agg_map.items()):
        x = df["step"].to_numpy()
        y = df[f"{metric_key}_mean"].to_numpy()
        plt.plot(x, y, marker="o", label=sampler_label)

        if plot_std:
            s = df[f"{metric_key}_std"].fillna(0.0).to_numpy()
            plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("training step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    return torch.load(path, map_location=device, weights_only=False)


def process_run(
    run_dir: Path,
    sampler_arg: str,
    n_repeats: int,
    seed: int,
    device: torch.device,
    output_root: Path,
    max_checkpoints: Optional[int],
    dry_run: bool,
) -> Optional[pd.DataFrame]:
    numeric_ckpts = discover_numeric_checkpoints(run_dir)
    if not numeric_ckpts:
        return None

    if max_checkpoints is not None:
        numeric_ckpts = numeric_ckpts[: max(0, max_checkpoints)]

    if dry_run:
        numeric_ckpts = numeric_ckpts[:1]

    first_step, first_path = numeric_ckpts[0]
    first_ckpt = load_checkpoint(first_path, device)

    base_graph = build_graph_from_checkpoint(first_ckpt, device)
    input_dim, output_dim = infer_model_dims(first_ckpt, base_graph)

    cfg_raw = first_ckpt["cfg"]
    cfg_eval = override_sampler_in_cfg(cfg_raw, sampler_arg)
    sampler_label = str(cfg_eval.sampling.type)

    model = create_inr_instance(cfg_eval, input_dim=input_dim, output_dim=output_dim, device=str(device))

    rows: List[Dict[str, Any]] = []
    total_ckpts = len(numeric_ckpts)
    print(
        f"[RUN] {run_dir.name}: checkpoints={total_ckpts}, repeats={n_repeats}, sampler={sampler_label}",
        flush=True,
    )

    for ckpt_idx, (step, ckpt_path) in enumerate(numeric_ckpts, start=1):
        ckpt_start = time.time()
        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["inr"])  # same architecture implied by training cfg

        gt_grad = full_graph_gradient(model, base_graph)

        # Reuse one sampler object per checkpoint so adaptive samplers can reuse
        # their internal cached grid/cell properties across repeat draws.
        sampler = create_inr_sampler(
            cfg_eval,
            model,
            base_graph,
            current_date_str="analysis_",
            run_name="grad_trend",
            device=str(device),
        )

        repeats: List[RepeatMetrics] = []
        for ridx in range(n_repeats):
            repeat_seed = seed + ridx
            repeat_metrics = single_repeat_metrics(
                model=model,
                base_graph=base_graph,
                step=step,
                repeat_seed=repeat_seed,
                sampler=sampler,
            )
            repeats.append(repeat_metrics)

        ckpt_stats = checkpoint_stats_from_repeats(repeats, step, gt_grad)
        row = checkpoint_metrics_to_row(ckpt_stats)
        row["run_name"] = run_dir.name
        row["sampler"] = sampler_label
        rows.append(row)

        elapsed = time.time() - ckpt_start
        print(
            f"[PROGRESS] {run_dir.name} | {ckpt_idx}/{total_ckpts} | step={step} | "
            f"grad_var={ckpt_stats.grad_variance:.4e} | corr={ckpt_stats.avg_pairwise_cosine:.4f} | "
            f"ortho={ckpt_stats.orthogonal_grad_error_mean:.4e} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

    if not rows:
        return None

    run_df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

    per_run_dir = output_root / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(per_run_dir / f"{run_dir.name}.csv", index=False)

    return run_df


def main() -> None:
    args = parse_args()

    checkpoint_parent = Path(args.checkpoint_parent)
    if not checkpoint_parent.exists():
        raise FileNotFoundError(f"checkpoint parent does not exist: {checkpoint_parent}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    run_dirs = discover_run_dirs(checkpoint_parent, args.run_name_filter)
    if not run_dirs:
        raise RuntimeError(f"No run directories found under: {checkpoint_parent}")

    if args.dry_run:
        run_dirs = run_dirs[:1]

    all_runs: List[pd.DataFrame] = []
    skipped_runs: List[str] = []

    for run_dir in run_dirs:
        try:
            run_df = process_run(
                run_dir=run_dir,
                sampler_arg=args.sampler,
                n_repeats=args.n_repeats,
                seed=args.seed,
                device=device,
                output_root=output_root,
                max_checkpoints=args.max_checkpoints,
                dry_run=args.dry_run,
            )
            if run_df is None:
                skipped_runs.append(run_dir.name)
                continue
            all_runs.append(run_df)
            print(
                f"[OK] {run_dir.name}: {len(run_df)} checkpoints processed, "
                f"sampler={run_df['sampler'].iloc[0]}"
            )
        except Exception as exc:
            skipped_runs.append(run_dir.name)
            print(f"[SKIP] {run_dir.name}: {exc}")

    if not all_runs:
        raise RuntimeError("No run produced metrics; check logs and input paths.")

    all_df = pd.concat(all_runs, ignore_index=True)
    all_df.to_csv(output_root / "all_runs_metrics.csv", index=False)

    agg_map: Dict[str, pd.DataFrame] = {}
    agg_dir = output_root / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    for sampler, df_sampler in all_df.groupby("sampler"):
        agg_df = aggregate_by_step(df_sampler)
        agg_map[str(sampler)] = agg_df
        agg_df.to_csv(agg_dir / f"sampler_{safe_sampler_label(str(sampler))}.csv", index=False)

    plots_dir = output_root / "plots"
    plot_sampler_metric(
        agg_map=agg_map,
        metric_key="grad_variance",
        ylabel="gradient variance",
        output_path=plots_dir / "gradient_variance_vs_step.png",
        plot_std=args.plot_std,
    )
    plot_sampler_metric(
        agg_map=agg_map,
        metric_key="avg_pairwise_cosine",
        ylabel="average pairwise cosine correlation",
        output_path=plots_dir / "gradient_correlation_vs_step.png",
        plot_std=args.plot_std,
    )
    plot_sampler_metric(
        agg_map=agg_map,
        metric_key="orthogonal_grad_error",
        ylabel="orthogonal gradient error",
        output_path=plots_dir / "orthogonal_gradient_error_vs_step.png",
        plot_std=args.plot_std,
    )

    summary = {
        "checkpoint_parent": str(checkpoint_parent),
        "sampler_arg": args.sampler,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "device": str(device),
        "run_dirs_considered": [p.name for p in run_dirs],
        "runs_processed": sorted(all_df["run_name"].unique().tolist()),
        "samplers_processed": sorted(all_df["sampler"].unique().tolist()),
        "skipped_runs": skipped_runs,
        "output_dir": str(output_root),
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Wrote analysis outputs to:", output_root)


if __name__ == "__main__":
    main()
