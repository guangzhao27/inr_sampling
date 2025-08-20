#!/bin/bash
#SBATCH -p csi
#SBATCH -t 01:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1

w0=13
sampling_rate=0.2
train_ratio=1
inner_steps=6
lr=5.6e-5 # 5.6e-5 non full 
depth=6
n_start=100
n_finish=4000

# null NMT random 2d_cluster_slic 2d_cluster_grid EVOS
for sample_type in EVOS
do
    run_name="SCENT_single_${sample_type}_${sampling_rate}_lr_${lr}_depth_${depth}_end_${n_finish}"

    cd /sdcc/u/smccue/projects/inr_sampling
    # source ~/anaconda3/etc/profile.d/conda.sh
    eval "$(conda shell.bash hook)"
    conda activate inr_sample
    # original lr = 5.6e-5
    python /sdcc/u/smccue/projects/inr_sampling/inr_sample/single_image_inr.py \
        data.dataset_name=NS \
        inr.model_type=siren \
        data.space_factor=1 \
        optim.batch_size=2 \
        optim.lr_inr=$lr \
        optim.epochs=5000 \
        optim.inner_steps=$inner_steps \
        inr.latent_dim=256 \
        inr.depth=$depth \
        inr.hidden_dim=155 \
        saved_checkpoint=False \
        wandb.name=$run_name \
        wandb.use_wandb=True \
        wandb.project=full-depth-grad \
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
        data.data_path=/sdcc/u/smccue/projects/inr_sampling/pde_bench/ns_incom_inhom_2d_512-0.h5 \
        data.data_type=hdf5 \
        data.single_time_frame=113
done

# Refer to inr_sample.yaml for choices for evos settings
# /sdcc/u/smccue/projects/inr_sampling/scent_data/1k
# /sdcc/u/smccue/projects/inr_sampling/pde_bench/ns_incom_inhom_2d_512-0.h5
# single_time_frame values:
# 800 143 585 113
# Chosen trajectory can be found in unstructure_dataset.py as variable "chosen_N"
