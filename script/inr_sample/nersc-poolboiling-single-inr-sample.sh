#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m2956_g

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchgeo

repo_root=/pscratch/sd/g/gzhao27/INR/INR_SAMPLE
cd "$repo_root"

w0=30
sampling_rate=1e-3
train_ratio=1
inner_steps=6
lr=1e-4
depth=6
n_start=11
n_finish=128
condition=100
field_key=temperature

# sampling choices: null random NMT 2d_grid_linear EVOS
for time_frame in 40 80 120 160 200; do
  for sample_type in random NMT 2d_grid_linear EVOS; do
    run_name="PoolBoiling2D_single_${sample_type}_Twall_${condition}_sampling_${sampling_rate}_lr_${lr}_depth_${depth}_t${time_frame}"

    python "$repo_root/inr_sample/single_image_inr.py" \
      data.dataset_name=PoolBoiling2D \
      data.data_path="$repo_root/data/PoolBoiling-SubCooled-FC72-2D" \
      data.poolboiling_condition=$condition \
      data.poolboiling_key=$field_key \
      data.poolboiling_sample_idx=0 \
      inr.model_type=siren \
      data.space_factor=1 \
      optim.batch_size=2 \
      optim.lr_inr=$lr \
      optim.epochs=2000 \
      optim.inner_steps=$inner_steps \
      optim.evo_every_epochs=200 \
      inr.latent_dim=256 \
      inr.depth=$depth \
      inr.hidden_dim=155 \
      inr.w0=$w0 \
      saved_checkpoint=False \
      wandb.name=$run_name \
      wandb.use_wandb=True \
      wandb.project=workshop-inr-sampling-revise \
      sampling.rate=$sampling_rate \
      sampling.type=$sample_type \
      sampling.sample_num_schedular=constant \
      sampling.mutation_method=constant \
      sampling.profile_interval_method=lin_dec \
      sampling.profile_guide=value \
      sampling.n_clusters_2d_end=$n_finish \
      sampling.n_clusters_2d_start=$n_start \
      "data.split_ratios=[${train_ratio}, 0.01, 0.01]" \
      data.data_type=other \
      data.single_time_frame=${time_frame}
  done
done

lr=5e-4
for time_frame in 40 80 120 160 200; do
  for sample_type in null; do
    run_name="PoolBoiling2D_single_${sample_type}_Twall_${condition}_sampling_${sampling_rate}_lr_${lr}_depth_${depth}_t${time_frame}"
    python "$repo_root/inr_sample/single_image_inr.py" \
      data.dataset_name=PoolBoiling2D \
      data.data_path="$repo_root/data/PoolBoiling-SubCooled-FC72-2D" \
      data.poolboiling_condition=$condition \
      data.poolboiling_key=$field_key \
      data.poolboiling_sample_idx=0 \
      inr.model_type=siren \
      data.space_factor=1 \
      optim.batch_size=2 \
      optim.lr_inr=$lr \
      optim.epochs=2000 \
      optim.inner_steps=$inner_steps \
      optim.evo_every_epochs=50 \
      inr.latent_dim=256 \
      inr.depth=$depth \
      inr.hidden_dim=155 \
      inr.w0=$w0 \
      saved_checkpoint=False \
      wandb.name=$run_name \
      wandb.use_wandb=True \
      wandb.project=workshop-inr-sampling-revise \
      sampling.rate=$sampling_rate \
      sampling.type=$sample_type \
      sampling.sample_num_schedular=constant \
      sampling.mutation_method=constant \
      sampling.profile_interval_method=lin_dec \
      sampling.profile_guide=value \
      sampling.n_clusters_2d_end=$n_finish \
      sampling.n_clusters_2d_start=$n_start \
      "data.split_ratios=[${train_ratio}, 0.01, 0.01]" \
      data.data_type=other \
      data.single_time_frame=${time_frame}
  done
done