#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m2956_g

#!/bin/bash

w0=13
sampling_rate=0.01
train_ratio=0.01
inner_steps=6
# null NMT random random 2d_cluster_slic 2d_cluster_grid
for sample_type in random
do
    run_name="SCENT_single_${sample_type}_${sampling_rate}_w0_${w0}_4layer"

    cd /pscratch/sd/g/gzhao27/INR/SOMA/
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate torchgeo

    python /pscratch/sd/g/gzhao27/INR/SOMA/inr_sample/single_image_inr.py \
        data.dataset_name=NS \
        inr.model_type=siren \
        data.space_factor=1 \
        optim.batch_size=2 \
        optim.lr_inr=5.6e-5 \
        optim.epochs=500 \
        optim.inner_steps=$inner_steps \
        inr.latent_dim=256 \
        inr.depth=6 \
        inr.hidden_dim=155 \
        saved_checkpoint=False \
        wandb.name=$run_name \
        wandb.use_wandb=True \
        wandb.project=inr-sampling-single-image \
        inr.w0=$w0 \
        sampling.rate=$sampling_rate \
        sampling.type=$sample_type \
        "data.split_ratios=[${train_ratio}, 0.01, 0.01]" \
        data.data_path=/pscratch/sd/g/gzhao27/INR/data/scent_data/1k \
        data.data_type=mmap
done
