#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m2956_g

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$SCRIPT_DIR"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchgeo

python script/inr_sample/compare_four_runs_gradient_trend.py \
  --checkpoint-parent Results/checkpoints \
  --output-dir Results/gradient_trend_comparison_four_runs_orthogonal_error \
  --n-repeats 4