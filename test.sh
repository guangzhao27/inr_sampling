#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$SCRIPT_DIR"

HOST=$(hostname -f)
if [[ $HOST == *"bnl"* ]]; then
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
  --n-repeats 4