#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m2956_g

w0=10
sampling_rate=1.0
sample_type=random
run_name=KS_${sample_type}_${sampling_rate}_w0_$w0_0.3


cd /pscratch/sd/g/gzhao27/INR/SOMA/
conda activate coral
python /pscratch/sd/g/gzhao27/INR/SOMA/inr_sample/inr_sample.py \
    data.dataset_name=NS \
    inr.model_type=siren \
    data.space_factor=1 \
    optim.batch_size=2 \
    optim.lr_inr=1e-4 \
    optim.epochs=500 \
    inr.latent_dim=128 \
    inr.depth=3 \
    inr.hidden_dim=32 \
    saved_checkpoint=False \
    wandb.name=$run_name \
    wandb.use_wandb=True \
    wandb.project=inr-sampling \
    inr.w0=$w0 \
    sampling.rate=$sampling_rate \
    sampling.type=$sample_type \
    "data.split_ratios=[0.3, 0.1, 0.1]" \
    data.data_path=/pscratch/sd/g/gzhao27/INR/data/KuramotoSivashinsky/sample300_time31_space64.npy \
    data.data_type=npy
    