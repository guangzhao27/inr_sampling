#!/bin/bash

checkpoint_path="/pscratch/sd/g/gzhao27/INR/INR_SAMPLE/Results/checkpoints/2026-03-24-10-08-20NS1024_single_2d_grid_linear_dataset_Piecewise2D_sampling_2e-4_lr_1e-4_depth_6_t100/99.pt"

python script/generate_image.py "$checkpoint_path"