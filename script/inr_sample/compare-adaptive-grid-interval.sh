#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m2956_g

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

# Conda activation scripts may reference unset backup vars; temporarily relax nounset.
set +u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchgeo
set -u

# -----------------------------
# Baseline experiment settings
# -----------------------------
w0=30
sampling_rate=1e-3
train_ratio=1
inner_steps=6
lr=1e-4
depth=6
time_frame=100
re=10000
epochs=3000

# Keep these aligned with your prior adaptive runs.
data_path="/pscratch/sd/g/gzhao27/INR/INR_SAMPLE/data/NS2d/ns_data_res2048_re${re}_7.npy"

# Sweep knobs for adaptive sampler.
interval_list=(50 200)
update_values_each_step_list=(true false)

for interval in "${interval_list[@]}"; do
  for update_values_each_step in "${update_values_each_step_list[@]}"; do
    run_name="NS1024_single_2d_grid_adaptive_int${interval}_updateval${update_values_each_step}_re_${re}_sampling_${sampling_rate}_lr_${lr}_depth_${depth}_t${time_frame}"

    python inr_sample/single_image_inr.py \
      data.dataset_name=NS \
      inr.model_type=siren \
      data.space_factor=1 \
      optim.batch_size=2 \
      optim.lr_inr=$lr \
      optim.epochs=$epochs \
      optim.inner_steps=$inner_steps \
      optim.evo_every_epochs=100 \
      inr.latent_dim=256 \
      inr.depth=$depth \
      inr.hidden_dim=155 \
      saved_checkpoint=False \
      wandb.name=$run_name \
      wandb.use_wandb=True \
      wandb.project=workshop-inr-sampling-revise \
      inr.w0=$w0 \
      sampling.type=2d_grid_adaptive \
      sampling.rate=$sampling_rate \
      sampling.adaptive_mode=loss \
      sampling.adaptive_grid_update_interval=$interval \
      sampling.adaptive_update_values_each_step=$update_values_each_step \
      "data.split_ratios=[${train_ratio}, 0.01, 0.01]" \
      data.data_path=$data_path \
      data.data_type=other \
      data.single_time_frame=${time_frame}
  done
done
