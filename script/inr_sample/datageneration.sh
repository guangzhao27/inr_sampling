#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m2956_g

cd /pscratch/sd/g/gzhao27/INR/INR_SAMPLE
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchgeo

python /pscratch/sd/g/gzhao27/INR/INR_SAMPLE/utils/data/generate_burgers2d.py \
  --H 1024 --W 1024 --T_steps 100 \
  --dt 0.001 --substeps 100 \
  --N_train_nu 10 \
  --out_dir /pscratch/sd/g/gzhao27/INR/data/2D_Burgers_1024