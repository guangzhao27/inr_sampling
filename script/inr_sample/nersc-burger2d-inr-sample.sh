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

cd "$REPO_ROOT"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchgeo

w0=30
sampling_rate=2e-4
train_ratio=1
inner_steps=6
lr=1e-4 # 5.6e-5 non full 
depth=6
n_start=11
n_finish=128
sample_type=NMT
time_frame=200
dataset_name=Burgers2D
data_path=/pscratch/sd/g/gzhao27/INR/data/2D_Burgers_Sols_Nu0.001.hdf5
burgers2d_sample_idx=0
# dataset name choices: Piecewise2D, NS2d, NS2d_inhom, NS2d_incom, NS2d_incom_inhom

# null NMT random 2d_cluster_slic 2d_grid_linear EVOS
# for time_frame in 100 120 140 160 180 200; do
#   for sample_type in NMT random 2d_grid_linear EVOS; do
for time_frame in 25; do
  for sample_type in NMT random 2d_grid_linear EVOS; do
    run_name="dataset_${dataset_name}_sample_${burgers2d_sample_idx}_${sample_type}_sampling_${sampling_rate}_lr_${lr}_depth_${depth}_t${time_frame}"
    python inr_sample/single_image_inr.py \
        data.dataset_name=$dataset_name \
      data.data_path=$data_path \
        data.burgers2d_sample_idx=$burgers2d_sample_idx \
        inr.model_type=siren \
        data.space_factor=1 \
        optim.batch_size=2 \
        optim.lr_inr=$lr \
        optim.epochs=1000 \
        optim.inner_steps=$inner_steps \
        inr.latent_dim=256 \
        inr.depth=$depth \
        inr.hidden_dim=155 \
        optim.evo_every_epochs=100 \
        saved_checkpoint=False \
        wandb.name=$run_name \
        wandb.use_wandb=True \
        wandb.project=workshop-inr-sampling-revise  \
        inr.w0=$w0 \
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

lr=1e-3

# data_type: mmap or other


        # data.data_path=/sdcc/u/smccue/projects/inr_sampling/pde_bench/ns_incom_inhom_2d_512-0.h5 \
        # data.data_type=hdf5 \
# Refer to inr_sample.yaml for choices for evos settings
# /sdcc/u/smccue/projects/inr_sampling/scent_data/1k
# /sdcc/u/smccue/projects/inr_sampling/pde_bench/ns_incom_inhom_2d_512-0.h5
# single_time_frame values:
# 800 143 585 113
# Chosen trajectory can be found in unstructure_dataset.py as variable "chosen_N"


# First test 
# first test the best lr for full
# test the best 