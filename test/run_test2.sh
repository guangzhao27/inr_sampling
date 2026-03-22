#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m2956_g

cd /pscratch/sd/g/gzhao27/INR/INR_SAMPLE/test
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchgeo

python -u /pscratch/sd/g/gzhao27/INR/INR_SAMPLE/test/grad_estimation.py