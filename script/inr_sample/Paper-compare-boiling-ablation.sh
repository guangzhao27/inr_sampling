#!/bin/bash
#SBATCH -p csi
#SBATCH -t 04:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1

# INSTRUCTION: Each script should start with:
#   conda activate /sdcc/u/gzhao/scratch/conda/inr_sampling

model_type="siren" # single_image_fourier_mlp or siren
method_list=(
  adaptive_with_no_area
)
w0=30
sampling_rate=2e-3
train_ratio=1
inner_steps=6
depth=3
n_start=20
n_finish=20
re=10000
condition=100 # for pool boiling
poolboiling_key="temperature"
#data_path="/sdcc/u/gzhao/scratch/inr_sampling/data/NS2d/ns_data_res2048_re${re}_7.npy"
data_path="/sdcc/u/gzhao/scratch/inr_sampling/data/PoolBoiling-SubCooled-FC72-2D"
problem_name="PoolBoiling2D_single_" # NS1024 or PoolBoiling2D_single_

if [[ "$problem_name" == PoolBoiling2D* ]]; then
  dataset_name="PoolBoiling2D"
else
  dataset_name="NS"
fi

# Usage:
#   bash compare_sgd-IC2.sh [test]
# Examples:
#   bash compare_sgd-IC2.sh true   # short test settings
#   bash compare_sgd-IC2.sh false  # paper settings (matches Paper-compare-NS1024.sh)
test_mode="${1:-false}"

if [[ "$test_mode" == "true" ]]; then
  project_name="test_runing"
  time_frames=(100)
  lr_list=(1e-2 3e-3 1e-3)
  default_epochs=5000
  full_epochs=250
  random_epochs=5000
elif [[ "$test_mode" == "false" ]]; then
  project_name="Bubble-inr-ablation"
  time_frames=(30 70)
  lr_list=(1e-3)
  default_epochs=5000
  full_epochs=2500
  random_epochs=10000
else
  echo "Invalid test option: $test_mode"
  echo "Expected: true or false"
  exit 1
fi




optimizer_name=sgd
sgd_momentum=0.9
sgd_nesterov=True
sgd_weight_decay=0.0
sgd_dampening=0.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HOST=$(hostname -f)
is_bnl=false
if [[ $HOST == *"bnl"* ]]; then
  is_bnl=true
  source ~/.bashrc
  conda activate /sdcc/u/gzhao/scratch/conda/inr_sampling
  wandb offline
  wandb_base_dir="${WANDB_DIR:-${REPO_ROOT}/coral/wandb}"
  wandb_run_dir="$wandb_base_dir/wandb"
  offline_log_file="/sdcc/u/gzhao/scratch/inr_sampling/coral/wandb/offline_run_paths.txt"
  command_log_file="/sdcc/u/gzhao/scratch/inr_sampling/offline_command_paths.sh"
  mkdir -p "$wandb_run_dir"
  touch "$offline_log_file"
  touch "$command_log_file"
else
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate torchgeo
  if [[ "$dataset_name" == "NS" ]]; then
    data_path="/pscratch/sd/g/gzhao27/INR/INR_SAMPLE/data/NS2d/ns_data_res2048_re${re}_7.npy"
  fi
fi

# LR sweep values for SGD
# methods: full random NMT grid_linear EVOS adaptive_topk_none adaptive_loss_sqrt_std adaptive_unbiased
for time_frame in "${time_frames[@]}"; do
for lr in "${lr_list[@]}"; do
  for case_name in "${method_list[@]}"; do

    # --- default values for all adaptive/sampling params ---
    sample_type="null"
    adaptive_mode="loss"
    adaptive_iterations=4
    adaptive_equal_cell_topk="False"
    adaptive_equal_cell_topk_count_mode="proportional"
    adaptive_equal_cell_topk_weight_mode="none"
    adaptive_weight_mode="none"
    power_for_loss_as_weight=1.0
    adaptive_grid_update_interval=100

    # --- per-method overrides ---
    epochs=$default_epochs
    if [[ "$case_name" == "full" ]]; then
      sample_type="null"
      epochs=$full_epochs

    elif [[ "$case_name" == "random" ]]; then
      sample_type="random"
      epochs=$random_epochs

    elif [[ "$case_name" == "NMT" ]]; then
      sample_type="NMT"

    elif [[ "$case_name" == "grid_linear" ]]; then
      sample_type="2d_grid_linear"

    elif [[ "$case_name" == "EVOS" ]]; then
      sample_type="EVOS"

    elif [[ "$case_name" == "adaptive_topk_none" ]]; then
      sample_type="2d_grid_adaptive"
      adaptive_equal_cell_topk="True"
      adaptive_weight_mode="none"
      adaptive_equal_cell_topk_weight_mode="none"

    elif [[ "$case_name" == "adaptive_loss_sqrt_std" ]]; then
      sample_type="2d_grid_adaptive"
      adaptive_equal_cell_topk="True"
      adaptive_weight_mode="none"
      adaptive_iterations=8
      adaptive_equal_cell_topk_count_mode="same"
      adaptive_equal_cell_topk_weight_mode="none"
      adaptive_mode="loss_sqrt_std"

    elif [[ "$case_name" == "adaptive_unbiased" ]]; then
      sample_type="2d_grid_adaptive"
      adaptive_mode="loss_sqrt_std"
      adaptive_iterations=8
      adaptive_equal_cell_topk="True"
      adaptive_equal_cell_topk_count_mode="same"
      adaptive_equal_cell_topk_weight_mode="area_over_count"
      adaptive_weight_mode="area_over_count"
      power_for_loss_as_weight=1.0
    
    elif [[ "$case_name" == "adaptive_best" ]]; then
      sample_type="2d_grid_adaptive"
      adaptive_mode="loss_sqrt_std"
      adaptive_iterations=8
      adaptive_equal_cell_topk="True"
      adaptive_equal_cell_topk_count_mode="same"
      adaptive_equal_cell_topk_weight_mode="loss_sqrt"
      adaptive_weight_mode="none"
      power_for_loss_as_weight=0.25

    elif [[ "$case_name" == "adaptive_large_bias" ]]; then
      sample_type="2d_grid_adaptive"
      adaptive_mode="loss_sqrt_std"
      adaptive_iterations=8
      adaptive_equal_cell_topk="True"
      adaptive_equal_cell_topk_count_mode="same"
      adaptive_equal_cell_topk_weight_mode="loss_sqrt"
      adaptive_weight_mode="none"
      power_for_loss_as_weight=1.0
    
    elif [[ "$case_name" == "adaptive_best_fast" ]]; then
      sample_type="2d_grid_adaptive"
      adaptive_mode="loss_sqrt_std"
      adaptive_iterations=4
      adaptive_equal_cell_topk="True"
      adaptive_equal_cell_topk_count_mode="same"
      adaptive_equal_cell_topk_weight_mode="loss_sqrt"
      adaptive_weight_mode="none"
      power_for_loss_as_weight=0.25
      adaptive_grid_update_interval=500
    
    elif [[ "$case_name" == "adaptive_with_no_area" ]]; then
      sample_type="2d_grid_adaptive"
      adaptive_mode="loss_sqrt_std"
      adaptive_iterations=4
      adaptive_equal_cell_topk="True"
      adaptive_equal_cell_topk_count_mode="same"
      adaptive_equal_cell_topk_weight_mode="loss_sqrt"
      adaptive_weight_mode="none"
      power_for_loss_as_weight=0.25
      adaptive_grid_update_interval=500

    else
      echo "Unknown case: $case_name"
      continue
    fi

    run_name="${problem_name}_${model_type}_${case_name}_re_${re}_sampling_${sampling_rate}_lr_${lr}_depth_${depth}_t${time_frame}"

    if [[ "$is_bnl" == "true" ]]; then
      before_runs="$(find "$wandb_run_dir" -maxdepth 1 -type d -name 'offline-run-*' -printf '%f\n' | sort)"
    fi

    python inr_sample/single_image_inr.py \
      data.dataset_name=$dataset_name \
      data.poolboiling_condition=$condition \
      data.poolboiling_key=$poolboiling_key \
      data.poolboiling_sample_idx=0 \
        inr.model_type=$model_type \
        data.space_factor=1 \
        optim.batch_size=2 \
        optim.optimizer=$optimizer_name \
        optim.sgd_momentum=$sgd_momentum \
        optim.sgd_nesterov=$sgd_nesterov \
        optim.sgd_weight_decay=$sgd_weight_decay \
        optim.sgd_dampening=$sgd_dampening \
        optim.lr_inr=$lr \
        optim.epochs=$epochs \
        optim.inner_steps=$inner_steps \
        optim.evo_every_epochs=100 \
        inr.latent_dim=256 \
        inr.depth=$depth \
        inr.hidden_dim=155 \
        inr.w0=$w0 \
        save_checkpoints=False \
        wandb.name=$run_name \
        wandb.use_wandb=True \
        wandb.project=$project_name \
        sampling.rate=$sampling_rate \
        sampling.type=$sample_type \
        sampling.adaptive_mode=$adaptive_mode \
        sampling.adaptive_iterations=$adaptive_iterations \
        sampling.adaptive_equal_cell_topk=$adaptive_equal_cell_topk \
        sampling.adaptive_equal_cell_topk_count_mode=$adaptive_equal_cell_topk_count_mode \
        sampling.adaptive_equal_cell_topk_weight_mode=$adaptive_equal_cell_topk_weight_mode \
        sampling.adaptive_weight_mode=$adaptive_weight_mode \
        sampling.power_for_loss_as_weight=$power_for_loss_as_weight \
        sampling.adaptive_weight_value_eps=1e-6 \
        sampling.adaptive_weight_clip_ratio=10 \
        sampling.sample_num_schedular=constant \
        sampling.mutation_method=constant \
        sampling.profile_interval_method=lin_dec \
        sampling.profile_guide=value \
        sampling.n_clusters_2d_start=$n_start \
        sampling.n_clusters_2d_end=$n_finish \
        sampling.adaptive_grid_update_interval=$adaptive_grid_update_interval \
        "data.split_ratios=[${train_ratio}, 0.01, 0.01]" \
        data.data_path=$data_path \
        data.data_type=other \
        data.single_time_frame=${time_frame}

    if [[ "$is_bnl" == "true" ]]; then
      after_runs="$(find "$wandb_run_dir" -maxdepth 1 -type d -name 'offline-run-*' -printf '%f\n' | sort)"
      new_runs="$(comm -13 <(printf "%s\n" "$before_runs") <(printf "%s\n" "$after_runs") || true)"
      if [[ -n "$new_runs" ]]; then
        while IFS= read -r run_dir_name; do
          [[ -z "$run_dir_name" ]] && continue
          printf "%s\t%s\t%s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$run_name" "$wandb_run_dir/$run_dir_name" >> "$offline_log_file"
          echo "echo '[SYNC]' $run_name; wandb sync $wandb_run_dir/$run_dir_name || echo '[FAILED]' $run_name" >> "$command_log_file"
        done <<< "$new_runs"
      else
        printf "%s\t%s\t%s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$run_name" "OFFLINE_DIR_NOT_FOUND" >> "$offline_log_file"
      fi
    fi

  done
done
done