#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g

w0=30
sampling_rate=0.001
train_ratio=1
inner_steps=6
lr=1e-4 # 5.6e-5 non full 
depth=6
n_start=11
n_finish=128
sample_type=NMT
time_frame=200

cd /pscratch/sd/g/gzhao27/INR/SOMA/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchgeo

# null NMT random 2d_grid_linear EVOS 2d_grid_adaptive.    2d_cluster_slic
for time_frame in 100; do
  for sample_type in 2d_grid_adaptive; do
    run_name="NS1024_single_${sample_type}_${sampling_rate}_lr_${lr}_depth_${depth}_end_${n_finish}_t${time_frame}vcount_lossfix"

    python /pscratch/sd/g/gzhao27/INR/INR_SAMPLE/inr_sample/single_image_inr.py \
        data.dataset_name=NS \
        inr.model_type=siren \
        data.space_factor=1 \
        optim.batch_size=2 \
        optim.lr_inr=$lr \
        optim.epochs=9000 \
        optim.inner_steps=$inner_steps \
        inr.latent_dim=256 \
        inr.depth=$depth \
        inr.hidden_dim=155 \
        saved_checkpoint=False \
        wandb.name=$run_name \
        wandb.use_wandb=True \
        wandb.project=neurips-workshop-inr-sampling  \
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
        data.data_path=/pscratch/sd/g/gzhao27/INR/data/NS2d/ns_data_res2048_re1000_7.npy\
        data.data_type=other \
        data.single_time_frame=${time_frame}
  done
done


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