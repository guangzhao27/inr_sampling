#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m2956_g

w0=30
sampling_rate=2e-3
train_ratio=1
inner_steps=6
lr=1e-4
depth=6
n_start=11
n_finish=128
re=10000
time_frame=100
optimizer_name=adamw

# adaptive_best fixed settings
sample_type="2d_grid_adaptive"
adaptive_mode="loss_sqrt_std"
adaptive_iterations=8
adaptive_equal_cell_topk="True"
adaptive_equal_cell_topk_count_mode="same"
adaptive_equal_cell_topk_weight_mode="loss_sqrt"
adaptive_weight_mode="none"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd /pscratch/sd/g/gzhao27/INR/INR_SAMPLE

HOST=$(hostname -f)
is_bnl=false
if [[ $HOST == *"bnl"* ]]; then
  is_bnl=true
  source ~/.bashrc
  conda activate /sdcc/u/gzhao/scratch/conda/inr_sampling
  wandb offline
  wandb_base_dir="${WANDB_DIR:-${REPO_ROOT}/coral/wandb}"
  wandb_run_dir="$wandb_base_dir/wandb"
  offline_log_file="$wandb_base_dir/offline_run_paths.txt"
  mkdir -p "$wandb_run_dir"
  touch "$offline_log_file"
  data_path="/sdcc/u/gzhao/scratch/inr_sampling/data/NS2d/ns_data_res2048_re${re}_7.npy"
else
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate torchgeo
  data_path="/pscratch/sd/g/gzhao27/INR/INR_SAMPLE/data/NS2d/ns_data_res2048_re${re}_7.npy"
fi

# sweep power_for_loss_as_weight for adaptive_best configuration
for power in 0.0 0.1 0.25 0.5 0.75 1.0 2.0; do

  run_name="NS1024_adaptive_best_power_${power}_re_${re}_sampling_${sampling_rate}_lr_${lr}_depth_${depth}_t${time_frame}"

  if [[ "$is_bnl" == "true" ]]; then
    before_runs="$(find "$wandb_run_dir" -maxdepth 1 -type d -name 'offline-run-*' -printf '%f\n' | sort)"
  fi

  python inr_sample/single_image_inr.py \
      data.dataset_name=NS \
      inr.model_type=siren \
      data.space_factor=1 \
      optim.batch_size=2 \
      optim.optimizer=$optimizer_name \
      optim.lr_inr=$lr \
      optim.epochs=5000 \
      optim.inner_steps=$inner_steps \
      optim.evo_every_epochs=100 \
      inr.latent_dim=256 \
      inr.depth=$depth \
      inr.hidden_dim=155 \
      inr.w0=$w0 \
      saved_checkpoint=False \
      wandb.name=$run_name \
      wandb.use_wandb=True \
      wandb.project=workshop-inr-sampling-power-tune \
      sampling.rate=$sampling_rate \
      sampling.type=$sample_type \
      sampling.adaptive_mode=$adaptive_mode \
      sampling.adaptive_iterations=$adaptive_iterations \
      sampling.adaptive_equal_cell_topk=$adaptive_equal_cell_topk \
      sampling.adaptive_equal_cell_topk_count_mode=$adaptive_equal_cell_topk_count_mode \
      sampling.adaptive_equal_cell_topk_weight_mode=$adaptive_equal_cell_topk_weight_mode \
      sampling.adaptive_weight_mode=$adaptive_weight_mode \
      sampling.power_for_loss_as_weight=$power \
      sampling.adaptive_weight_value_eps=1e-6 \
      sampling.adaptive_weight_clip_ratio=10 \
      sampling.sample_num_schedular=constant \
      sampling.mutation_method=constant \
      sampling.profile_interval_method=lin_dec \
      sampling.profile_guide=value \
      sampling.n_clusters_2d_start=$n_start \
      sampling.n_clusters_2d_end=$n_finish \
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
      done <<< "$new_runs"
    else
      printf "%s\t%s\t%s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$run_name" "OFFLINE_DIR_NOT_FOUND" >> "$offline_log_file"
    fi
  fi

done
