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
sampling_rate=2e-3
train_ratio=1
inner_steps=6
lr=1e-4 # 5.6e-5 non full 
depth=6
n_start=11
n_finish=128
re=10000
data_path="/pscratch/sd/g/gzhao27/INR/INR_SAMPLE/data/NS2d/ns_data_res2048_re${re}_7.npy"
# null NMT random 2d_cluster_slic 2d_grid_linear EVOS
# # for time_frame in 100 120 140 160 180 200; do
# #   for sample_type in NMT random 2d_grid_linear EVOS; do
for time_frame in 100; do
  for case_name in \
    random \
    grid_linear \
    adaptive_topk_none \
    adaptive_topk_area_over_count; do

    if [[ "$case_name" == "random" ]]; then
      sample_type="random"
      adaptive_equal_cell_topk="False"
      adaptive_weight_mode="none"
      adaptive_equal_cell_topk_weight_mode="none"
    elif [[ "$case_name" == "grid_linear" ]]; then
      sample_type="2d_grid_linear"
      adaptive_equal_cell_topk="False"
      adaptive_weight_mode="none"
      adaptive_equal_cell_topk_weight_mode="none"
    elif [[ "$case_name" == "adaptive_topk_none" ]]; then
      sample_type="2d_grid_adaptive"
      adaptive_equal_cell_topk="True"
      adaptive_weight_mode="none"
      adaptive_equal_cell_topk_weight_mode="none"
    elif [[ "$case_name" == "adaptive_topk_area_over_count" ]]; then
      sample_type="2d_grid_adaptive"
      adaptive_equal_cell_topk="True"
      adaptive_weight_mode="area_over_count"
      adaptive_equal_cell_topk_weight_mode="area_over_count"
    else
      echo "Unknown case: $case_name"
      continue
    fi

    run_name="NS1024_single_${case_name}_re_${re}_sampling_${sampling_rate}_lr_${lr}_depth_${depth}_t${time_frame}"

    python inr_sample/single_image_inr.py \
        data.dataset_name=NS \
        inr.model_type=siren \
        data.space_factor=1 \
        optim.batch_size=2 \
        optim.lr_inr=$lr \
        optim.epochs=5000 \
        optim.inner_steps=$inner_steps \
        optim.evo_every_epochs=100 \
        inr.latent_dim=256 \
        inr.depth=$depth \
        inr.hidden_dim=155 \
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
        sampling.adaptive_equal_cell_topk=$adaptive_equal_cell_topk \
        sampling.adaptive_weight_mode=$adaptive_weight_mode \
        sampling.adaptive_equal_cell_topk_weight_mode=$adaptive_equal_cell_topk_weight_mode \
        sampling.adaptive_weight_value_eps=1e-6 \
        sampling.adaptive_weight_clip_ratio=10 \
        sampling.n_clusters_2d_end=$n_finish \
        sampling.n_clusters_2d_start=$n_start \
        "data.split_ratios=[${train_ratio}, 0.01, 0.01]" \
        data.data_path=$data_path \
        data.data_type=other \
        data.single_time_frame=${time_frame}
done
done
# lr=5e-4 # 5.6e-5 non full 
# for time_frame in 100 120 140 160 180 200; do
#   for sample_type in null; do
#     run_name="NS1024_single_${sample_type}_re_${re}_sampling_${sampling_rate}_lr_${lr}_depth_${depth}_t${time_frame}"

#     python inr_sample/single_image_inr.py \
#         data.dataset_name=NS \
#         inr.model_type=siren \
#         data.space_factor=1 \
#         optim.batch_size=2 \
#         optim.lr_inr=$lr \
#         optim.epochs=2000 \
#         optim.inner_steps=$inner_steps \
#         inr.latent_dim=256 \
#         inr.depth=$depth \
#         optim.evo_every_epochs=10 \
#         inr.hidden_dim=155 \
#         saved_checkpoint=False \
#         wandb.name=$run_name \
#         wandb.use_wandb=True \
#         wandb.project=workshop-inr-sampling-revise  \
#         inr.w0=$w0 \
#         sampling.rate=$sampling_rate \
#         sampling.type=$sample_type \
#         sampling.sample_num_schedular=constant \
#         sampling.mutation_method=constant \
#         sampling.profile_interval_method=lin_dec \
#         sampling.profile_guide=value \
#         sampling.n_clusters_2d_end=$n_finish \
#         sampling.n_clusters_2d_start=$n_start \
#         "data.split_ratios=[${train_ratio}, 0.01, 0.01]" \
#         data.data_path=$data_path \
#         data.data_type=other \
#         data.single_time_frame=${time_frame}
#   done
# done


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