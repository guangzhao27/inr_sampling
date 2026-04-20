#!/usr/bin/env python3
"""Compare gradient trend metrics for four specific experiment runs.

This script reuses the standalone analyzer functions from
`script/inr_sample/gradient_trend_analysis.py` and produces two plots:

1) gradient variance vs training step
2) gradient correlation vs training step

It keeps each run as an independent curve, which is important when multiple
runs share the same sampler type (e.g., two adaptive-topk variants).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from script.inr_sample import gradient_trend_analysis as gta


DEFAULT_CASE_FILTERS = {
    "random": "single_random_re_10000_sampling_2e-3_lr_1e-4_depth_6_t100",
    "2d_grid_linear": "single_grid_linear_re_10000_sampling_2e-3_lr_1e-4_depth_6_t100",
    "adaptive_topk_none": "single_adaptive_topk_none_re_10000_sampling_2e-3_lr_1e-4_depth_6_t100",
    "adaptive_topk_area_over_count": "single_adaptive_topk_area_over_count_re_10000_sampling_2e-3_lr_1e-4_depth_6_t100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 4 run cases with gradient-variance and gradient-correlation trend plots."
    )
    parser.add_argument(
        "--checkpoint-parent",
        type=str,
        default="Results/checkpoints",
        help="Parent folder containing run subdirectories with numeric-step checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Results/gradient_trend_comparison_four_runs",
        help="Output directory for per-run CSVs and comparison plots.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=4,
        help="Number of repeated samplings per checkpoint for gradient stats.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for repeated sampling; repeat k uses seed+k.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Optional cap on checkpoints per run (oldest-first).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only the first checkpoint from each case for quick checks.",
    )
    parser.add_argument(
        "--plot-std",
        action="store_true",
        help="If repeated runs are selected per case, add +/- std shading.",
    )
    parser.add_argument(
        "--case-random",
        type=str,
        default=DEFAULT_CASE_FILTERS["random"],
        help="Substring used to locate the random run directory.",
    )
    parser.add_argument(
        "--case-grid-linear",
        type=str,
        default=DEFAULT_CASE_FILTERS["2d_grid_linear"],
        help="Substring used to locate the 2d_grid_linear run directory.",
    )
    parser.add_argument(
        "--case-adaptive-none",
        type=str,
        default=DEFAULT_CASE_FILTERS["adaptive_topk_none"],
        help="Substring used to locate adaptive_topk_none run directory.",
    )
    parser.add_argument(
        "--case-adaptive-area-over-count",
        type=str,
        default=DEFAULT_CASE_FILTERS["adaptive_topk_area_over_count"],
        help="Substring used to locate adaptive_topk_area_over_count run directory.",
    )
    return parser.parse_args()


def _find_latest_run(run_dirs: List[Path], keyword: str) -> Optional[Path]:
    candidates = [p for p in run_dirs if keyword in p.name]
    if not candidates:
        return None
    # Timestamp prefix in folder names is lexicographically sortable.
    return sorted(candidates)[-1]


def _plot_case_metric(
    case_dfs: Dict[str, pd.DataFrame],
    metric_col: str,
    ylabel: str,
    output_path: Path,
    plot_std: bool = False,
    std_col: Optional[str] = None,
) -> None:
    plt.figure(figsize=(9, 5))
    for case_label, df in case_dfs.items():
        x = df["step"].to_numpy()
        y = df[metric_col].to_numpy()
        plt.plot(x, y, marker="o", label=case_label)

        if plot_std and std_col is not None and std_col in df.columns:
            s = df[std_col].fillna(0.0).to_numpy()
            plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.xlabel("training step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()

    checkpoint_parent = Path(args.checkpoint_parent)
    if not checkpoint_parent.exists():
        raise FileNotFoundError(f"checkpoint parent does not exist: {checkpoint_parent}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = gta.discover_run_dirs(checkpoint_parent, name_filter=None)
    if not run_dirs:
        raise RuntimeError(f"No run directories found under: {checkpoint_parent}")

    case_keywords = {
        "random": args.case_random,
        "2d_grid_linear": args.case_grid_linear,
        "adaptive_topk_none": args.case_adaptive_none,
        "adaptive_topk_area_over_count": args.case_adaptive_area_over_count,
    }

    selected_runs: Dict[str, Path] = {}
    for case_label, keyword in case_keywords.items():
        matched = _find_latest_run(run_dirs, keyword)
        if matched is None:
            raise RuntimeError(
                f"Could not find run folder for case '{case_label}' with keyword '{keyword}'."
            )
        selected_runs[case_label] = matched

    device = torch.device(args.device)
    case_dfs: Dict[str, pd.DataFrame] = {}
    skipped_cases: Dict[str, str] = {}

    per_run_root = output_dir / "per_run"
    per_run_root.mkdir(parents=True, exist_ok=True)

    for case_label, run_dir in selected_runs.items():
        try:
            run_df = gta.process_run(
                run_dir=run_dir,
                sampler_arg="auto",
                n_repeats=args.n_repeats,
                seed=args.seed,
                device=device,
                output_root=output_dir,
                max_checkpoints=args.max_checkpoints,
                dry_run=args.dry_run,
            )
            if run_df is None or run_df.empty:
                skipped_cases[case_label] = "no metrics produced"
                continue

            # Keep explicit case label for plotting/export.
            run_df = run_df.copy()
            run_df["case"] = case_label
            case_dfs[case_label] = run_df
            run_df.to_csv(per_run_root / f"{case_label}.csv", index=False)
            print(f"[OK] {case_label}: {run_dir.name} ({len(run_df)} checkpoints)")
        except Exception as exc:
            skipped_cases[case_label] = str(exc)
            print(f"[SKIP] {case_label}: {exc}")

    if len(case_dfs) == 0:
        raise RuntimeError("No case produced metrics; cannot generate comparison plots.")

    all_df = pd.concat(case_dfs.values(), ignore_index=True)
    all_df.to_csv(output_dir / "all_cases_metrics.csv", index=False)

    plots_dir = output_dir / "plots"
    _plot_case_metric(
        case_dfs=case_dfs,
        metric_col="grad_variance",
        ylabel="gradient variance",
        output_path=plots_dir / "gradient_variance_comparison.png",
        plot_std=False,
    )
    _plot_case_metric(
        case_dfs=case_dfs,
        metric_col="avg_pairwise_cosine",
        ylabel="average pairwise cosine correlation",
        output_path=plots_dir / "gradient_correlation_comparison.png",
        plot_std=False,
    )
    _plot_case_metric(
        case_dfs=case_dfs,
        metric_col="orthogonal_grad_error_mean",
        ylabel="orthogonal gradient error",
        output_path=plots_dir / "orthogonal_gradient_error_comparison.png",
        plot_std=args.plot_std,
        std_col="orthogonal_grad_error_std",
    )

    summary = {
        "checkpoint_parent": str(checkpoint_parent),
        "selected_runs": {k: str(v) for k, v in selected_runs.items()},
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "device": str(device),
        "cases_processed": sorted(case_dfs.keys()),
        "cases_skipped": skipped_cases,
        "output_dir": str(output_dir),
        "plots": {
            "gradient_variance": str(plots_dir / "gradient_variance_comparison.png"),
            "gradient_correlation": str(plots_dir / "gradient_correlation_comparison.png"),
            "orthogonal_gradient_error": str(plots_dir / "orthogonal_gradient_error_comparison.png"),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Wrote four-run comparison outputs to:", output_dir)


if __name__ == "__main__":
    main()
