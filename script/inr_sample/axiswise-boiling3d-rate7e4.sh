#!/bin/bash
#SBATCH -p csi
#SBATCH -t 12:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1
#SBATCH -J axiswise_boil3d
#SBATCH -o logs/axiswise_boil3d_%j.out

# Axis-wise (anisotropic) octree split on PoolBoiling3D, region-count-matched to
# the saved isotropic rate=7e-4 baseline (K_iso = 2500).
#
# Config is a bit-for-bit copy of the deleted `big-boiling-rate7e4-3seed.sh`
# (extracted from the saved offline W&B run
#  coral/wandb/wandb/offline-run-20260725_202422-9bt1duz0), changing ONLY the
# partition step:  sampling.split_mode=axiswise + sampling.target_num_regions=2500.
#
# Env vars (override on the command line):
#   SPLIT_MODE  axiswise | isotropic   (default axiswise)
#   EPOCHS      training epochs         (default 1200)
#   EVERY       eval interval           (default 20)
#   SEEDS       space-separated seeds   (default "42 43 44")
#   TAG         run_name prefix         (default axiswise_boiling3d)

source ~/.bashrc
conda activate /sdcc/u/gzhao/scratch/conda/inr_sampling
cd /hpcgpfs01/scratch/gzhao/inr_sampling
wandb offline

DATA=/sdcc/u/gzhao/scratch/inr_sampling/data/PoolBoiling-SubCooled-FC72-2D
SPLIT_MODE=${SPLIT_MODE:-axiswise}
EPOCHS=${EPOCHS:-1200}
EVERY=${EVERY:-20}
SEEDS=${SEEDS:-"42 43 44"}
TAG=${TAG:-axiswise_boiling3d}

common=(
  inr.model_type=siren inr.depth=10 inr.hidden_dim=384 inr.w0=60 inr.latent_dim=512
  data.dataset_name=PoolBoiling3D data.data_type=other
  data.data_path="$DATA"
  data.poolboiling_condition=100 data.poolboiling_key=temperature data.poolboiling_sample_idx=0
  data.space_factor=1 data.volume_time_start=58 data.volume_num_frames=64
  "data.split_ratios=[1,0.01,0.01]"
  optim.optimizer=sgd optim.sgd_nesterov=True optim.sgd_momentum=0.9
  optim.batch_size=2 optim.lr_inr=5e-3 optim.inner_steps=6
  optim.epochs="$EPOCHS" optim.evo_every_epochs="$EVERY"
  sampling.type=3d_grid_adaptive sampling.rate=7e-4 sampling.adaptive_mode=loss_sqrt_std
  sampling.adaptive_iterations=3 sampling.subdivision_percentage=10
  sampling.adaptive_grid_update_interval=100 sampling.adaptive_initial_grid_size=8
  sampling.adaptive_equal_cell_topk=False sampling.adaptive_equal_cell_topk_count_mode=same
  sampling.adaptive_equal_cell_topk_weight_mode=none sampling.adaptive_weight_mode=none
  sampling.split_mode="$SPLIT_MODE" sampling.target_num_regions=2500
  save_sampled_frames=False
  wandb.use_wandb=True
)

for seed in $SEEDS; do
  run_name="${TAG}_${SPLIT_MODE}_seed${seed}"
  echo "======================================================================"
  echo "=== $run_name  (epochs=$EPOCHS every=$EVERY) ==="
  echo "======================================================================"
  python inr_sample/single_image_inr.py \
    "${common[@]}" \
    data.seed="$seed" \
    wandb.name="$run_name"
  echo "=== done $run_name (exit $?) ==="
done
