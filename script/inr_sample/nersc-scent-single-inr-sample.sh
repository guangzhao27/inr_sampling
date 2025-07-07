#!/bin/bash
#SBATCH -p csi
#SBATCH -t 20:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1

w0=13
sampling_rate=0.1
train_ratio=1
inner_steps=6
# null NMT random random 2d_cluster_slic 2d_cluster_grid
for sample_type in null
do
    run_name="SCENT_single_${sample_type}_${sampling_rate}_w0_${w0}_4layer"

    cd /sdcc/u/smccue/projects/inr_sampling
    # source ~/anaconda3/etc/profile.d/conda.sh
    eval "$(conda shell.bash hook)"
    conda activate inr_sample

    python /sdcc/u/smccue/projects/inr_sampling/inr_sample/single_image_inr.py \
        data.dataset_name=NS \
        inr.model_type=siren \
        data.space_factor=1 \
        optim.batch_size=2 \
        optim.lr_inr=5.6e-5 \
        optim.epochs=5000 \
        optim.inner_steps=$inner_steps \
        inr.latent_dim=256 \
        inr.depth=3 \
        inr.hidden_dim=155 \
        saved_checkpoint=False \
        wandb.name=$run_name \
        wandb.use_wandb=True \
        wandb.project=full-depth-grad \
        inr.w0=$w0 \
        sampling.rate=$sampling_rate \
        sampling.type=$sample_type \
        "data.split_ratios=[${train_ratio}, 0.01, 0.01]" \
        data.data_path=/sdcc/u/smccue/projects/inr_sampling/scent_data/1k \
        data.data_type=mmap
done
