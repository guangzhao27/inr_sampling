#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchgeo

checkpoint_path="/pscratch/sd/g/gzhao27/INR/INR_SAMPLE/Results/checkpoints/2026-03-25-10-40-27dataset_Burgers2D_sample_0_null_sampling_1e-3_lr_1e-3_depth_6_t100/999.pt"

python /pscratch/sd/g/gzhao27/INR/INR_SAMPLE/script/generate_image.py "$checkpoint_path"