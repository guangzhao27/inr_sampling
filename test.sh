#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$SCRIPT_DIR"

HOST=$(hostname -f)
if [[ $HOST == *"bnl"* ]]; then
  echo hello world
  source ~/.bashrc
  conda activate /sdcc/u/gzhao/scratch/conda/inr_sampling
  wandb offline
else
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate torchgeo
fi

python script/inr_sample/compare_four_runs_gradient_trend.py \
  --checkpoint-parent Results/checkpoints \
  --output-dir Results/gradient_trend_comparison_four_runs_orthogonal_error \
  --case-random "single_random_re_10000_sampling_2e-3_lr_1e-4_depth_6_t100" \
  --case-grid-linear "single_random_re_10000_sampling_2e-3_lr_1e-4_depth_6_t100" \
  --case-adaptive-none "single_random_re_10000_sampling_2e-3_lr_1e-4_depth_6_t100" \
  --case-adaptive-area-over-count "single_random_re_10000_sampling_2e-3_lr_1e-4_depth_6_t100" \
  --n-repeats 4