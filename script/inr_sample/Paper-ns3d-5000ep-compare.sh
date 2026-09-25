#!/bin/bash
#SBATCH -p csi
#SBATCH -t 01:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1
#SBATCH --job-name=ns3d-5000ep
#SBATCH -o logs/ns3d-5000ep_%j.out

# NS3D's iter=5 vs Uniform at epochs=5000 (paper reports at 1500ep, where iter=5 vs
# iter=3 was within noise: 0.1934+/-0.0018 vs 0.1948+/-0.0022). This mirrors the
# PoolBoiling3D epoch-scaling check to see whether iter=5's (non-)advantage over
# Uniform looks different at a much longer training budget. Single seed=42 first pass.

REPO_ROOT="/hpcgpfs01/scratch/gzhao/inr_sampling"
source ~/.bashrc
conda activate /sdcc/u/gzhao/scratch/conda/inr_sampling
wandb offline
cd "$REPO_ROOT"

NPY=/sdcc/u/gzhao/scratch/inr_sampling/data/NS2d/ns_data_res2048_re10000_7.npy
SEED=42

common=(
  inr.model_type=siren inr.depth=10 inr.hidden_dim=384 inr.w0=60 inr.latent_dim=512
  data.dataset_name=NS3D data.data_type=other
  data.data_path="$NPY"
  data.space_factor=2 data.volume_time_start=100 data.volume_num_frames=64
  "data.split_ratios=[1,0.01,0.01]"
  optim.optimizer=sgd optim.sgd_nesterov=True optim.sgd_momentum=0.9
  optim.batch_size=2 optim.lr_inr=2e-3 optim.inner_steps=6
  optim.epochs=5000 optim.evo_every_epochs=20
  sampling.rate=7e-4
  save_sampled_frames=False
  data.seed=$SEED
  wandb.use_wandb=True wandb.project=NIPS-inr-sampling
)

echo "=========== uniform random, epochs=5000 ==========="
python inr_sample/single_image_inr.py "${common[@]}" \
    sampling.type=random \
    wandb.name="ns3d5000_random_seed${SEED}"
echo "DONE random exit=$?"

echo "=========== ACES iter=5 (candidate), epochs=5000 ==========="
python inr_sample/single_image_inr.py "${common[@]}" \
    sampling.type=3d_grid_adaptive sampling.adaptive_mode=loss_sqrt_std \
    sampling.adaptive_iterations=5 sampling.subdivision_percentage=10 \
    sampling.adaptive_grid_update_interval=100 sampling.adaptive_initial_grid_size=8 \
    sampling.adaptive_equal_cell_topk=False sampling.adaptive_equal_cell_topk_count_mode=same \
    sampling.adaptive_equal_cell_topk_weight_mode=none sampling.adaptive_weight_mode=none \
    wandb.name="ns3d5000_iter5_seed${SEED}"
echo "DONE iter5 exit=$?"

echo "NS3D_5000EP_COMPARE_DONE"
