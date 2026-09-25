#!/bin/bash
#SBATCH -p csi
#SBATCH -t 01:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1
#SBATCH --job-name=boil3d-5000ep
#SBATCH -o logs/boil3d-5000ep_%j.out

# PoolBoiling3D's ACES table (Experiment.tex) is reported at 1200 epochs, which the
# loss curve shows is NOT converged (still dropping ~3-4%/100ep at the cutoff -- see
# boil3dbase_iter5_seed42's history). This tests whether doubling to 2400 epochs
# changes the iter=3 (paper) vs iter=5 (candidate, Agent-report/rebuttal-setting.md)
# gap, and adds a matching Uniform random arm at the same epoch count (the paper's own
# Uniform number is also only reported at 1200ep). Single seed=42 first pass.

REPO_ROOT="/hpcgpfs01/scratch/gzhao/inr_sampling"
source ~/.bashrc
conda activate /sdcc/u/gzhao/scratch/conda/inr_sampling
wandb offline
cd "$REPO_ROOT"

DATA=/sdcc/u/gzhao/scratch/inr_sampling/data/PoolBoiling-SubCooled-FC72-2D
SEED=42

common=(
  inr.model_type=siren inr.depth=10 inr.hidden_dim=384 inr.w0=60 inr.latent_dim=512
  data.dataset_name=PoolBoiling3D data.data_type=other
  data.data_path="$DATA"
  data.poolboiling_condition=100 data.poolboiling_key=temperature data.poolboiling_sample_idx=0
  data.space_factor=1 data.volume_time_start=58 data.volume_num_frames=64
  "data.split_ratios=[1,0.01,0.01]"
  optim.optimizer=sgd optim.sgd_nesterov=True optim.sgd_momentum=0.9
  optim.batch_size=2 optim.lr_inr=5e-3 optim.inner_steps=6
  optim.epochs=5000 optim.evo_every_epochs=20
  sampling.rate=7e-4
  save_sampled_frames=False
  data.seed=$SEED
  wandb.use_wandb=True wandb.project=NIPS-inr-sampling
)

echo "=========== uniform random, epochs=2400 ==========="
python inr_sample/single_image_inr.py "${common[@]}" \
    sampling.type=random \
    wandb.name="boil3d5000_random_seed${SEED}"
echo "DONE random exit=$?"

echo "=========== ACES iter=3 (paper current), epochs=2400 ==========="
python inr_sample/single_image_inr.py "${common[@]}" \
    sampling.type=3d_grid_adaptive sampling.adaptive_mode=loss_sqrt_std \
    sampling.adaptive_iterations=3 sampling.subdivision_percentage=10 \
    sampling.adaptive_grid_update_interval=100 sampling.adaptive_initial_grid_size=8 \
    sampling.adaptive_equal_cell_topk=False sampling.adaptive_equal_cell_topk_count_mode=same \
    sampling.adaptive_equal_cell_topk_weight_mode=none sampling.adaptive_weight_mode=none \
    wandb.name="boil3d5000_iter3_seed${SEED}"
echo "DONE iter3 exit=$?"

echo "=========== ACES iter=5 (candidate), epochs=2400 ==========="
python inr_sample/single_image_inr.py "${common[@]}" \
    sampling.type=3d_grid_adaptive sampling.adaptive_mode=loss_sqrt_std \
    sampling.adaptive_iterations=5 sampling.subdivision_percentage=10 \
    sampling.adaptive_grid_update_interval=100 sampling.adaptive_initial_grid_size=8 \
    sampling.adaptive_equal_cell_topk=False sampling.adaptive_equal_cell_topk_count_mode=same \
    sampling.adaptive_equal_cell_topk_weight_mode=none sampling.adaptive_weight_mode=none \
    wandb.name="boil3d5000_iter5_seed${SEED}"
echo "DONE iter5 exit=$?"

echo "BOIL3D_5000EP_COMPARE_DONE"
