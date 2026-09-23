#!/bin/bash
#SBATCH -p csi
#SBATCH -t 06:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1
#SBATCH --job-name=ffn-s1
#SBATCH --output=slurm-ffn-s1-%j.out

# Fourier-feature-network (FFN) architecture-generality experiment, Stage 1.
#
# Reviewers objected that the paper tests one architecture. This replaces the SIREN
# backbone with a Gaussian-random-Fourier-feature ReLU MLP and re-runs the sampler
# comparison, finding each method's own best learning rate.
#
# Architecture is capacity-matched to the paper's SIREN, deliberately:
#   SIREN  depth=3 hidden=155                    -> 104,471 params
#   FFN    depth=3 hidden=155 mapping_size=256   -> 104,161 params
# so no gap between samplers can be blamed on one model being bigger.
#
# Every method gets its OWN lr sweep -- unlike the paper scripts, which share one lr per
# dataset. A ReLU MLP under SGD peaks nowhere near SIREN's, and a shared lr would silently
# handicap whichever method it suits least. All methods run 5000 epochs (the paper gives
# `random` 10000 for wall-clock matching; equal steps isolates the sampling strategy).
#
# These scripts deliberately do NOT write offline_run_paths.txt, so concurrent submission
# is safe; results come from scanning .wandb directly with extract_wandb_runs.py.
#
#   for m in random nmt evos strat aces; do
#     sbatch --export=ALL,DATASET=boil,METHOD=$m script/inr_sample/ffn-stage1-lr.sh
#     sbatch --export=ALL,DATASET=ns,METHOD=$m   script/inr_sample/ffn-stage1-lr.sh
#   done
#
# Budgets at rate 2e-3:
#   PoolBoiling2D   384^2  =  147456 pts -> n =  294/step ; stratified n_bins=8  ->  64 cells
#   NS1024         1024^2  = 1048576 pts -> n = 2097/step ; stratified n_bins=23 -> 529 cells
#
# ACES leaf count follows L <- L + 3*max(1, floor(L*pct/100)) from an 8x8 = 64 initial grid
# and does NOT depend on the dataset or on `rate` (initial_grid_size is hardcoded at
# SamplerWrapper.py:959):
#     iter |  pct=10 |  pct=20
#        5 |     226 |     652
#        6 |     292 |    1042
#        7 |     379 |    1666
#        8 |     490 |    2665
#       10 |     826 |    6820
# With FLOOR=min_one the batch silently exceeds budget once leaves > n, so leaf counts
# above the budget require FLOOR=soft (E[total] = n). `sampled_points` is logged every eval,
# so equal budget is verified from the runs rather than assumed.
#
# ⚠️ `save_sampled_frames=False` does NOT stop frame writing. single_image_inr.py:486 calls
# sample(save_image=True) on every eval step ungated by that flag, so random / NMT /
# stratified each dump ~50 MB of PNGs per run regardless. Only ACES escapes. Budget disk
# accordingly, or clean sampled_frames/ afterwards.

REPO_ROOT="/hpcgpfs01/scratch/gzhao/inr_sampling"
source ~/.bashrc
conda activate /sdcc/u/gzhao/scratch/conda/inr_sampling
export WANDB_MODE=offline
cd "$REPO_ROOT"

: "${DATASET:=boil}"
: "${METHOD:=random}"
: "${SEED:=42}"
: "${LRS:=1e-1 3e-2 1e-2 3e-3 1e-3 1e-4}"
: "${EPOCHS:=5000}"   # lowered only for the smoke test
: "${PROJECT:=NIPS-inr-ffn}"

# --- ACES partition knobs ---
: "${ACES_ITER:=5}"     # quadtree depth
: "${SUBDIV_PCT:=10}"   # growth rate; with ACES_ITER it sets the leaf count
: "${FLOOR:=min_one}"   # soft lifts the leaves<budget ceiling min_one imposes
: "${POWER:=1.0}"       # exponent of the loss-powered within-cell weighting

# --- stratified knob ---
: "${NBINS:=}"          # cells per axis; empty = the per-dataset default

# --- model-capacity knobs; defaults reproduce the capacity-matched architecture exactly ---
: "${FSCALE:=10.0}"     # FFN bandwidth: coords in [-1,1], so ~2*scale cycles across the field
: "${FMAP:=256}"
: "${HID:=155}"
: "${MODEL:=single_image_fourier_mlp}"   # MODEL=siren runs the paper backbone through this
: "${W0:=30}"                            # identical harness, for a controlled comparison

# Non-default settings are tagged into the run name so diagnostic arms can never be
# confused with the main protocol by the name-scanning extractor.
name_suffix=""
[[ "$MODEL" != "single_image_fourier_mlp" ]] && name_suffix="${name_suffix}_${MODEL}"
[[ "$ACES_ITER" != "5" && "$METHOD" == "aces" ]] && name_suffix="${name_suffix}_iter${ACES_ITER}"
[[ "$POWER" != "1.0" && "$METHOD" == "aces" ]] && name_suffix="${name_suffix}_pow${POWER}"
[[ "$SUBDIV_PCT" != "10" && "$METHOD" == "aces" ]] && name_suffix="${name_suffix}_pct${SUBDIV_PCT}"
[[ "$FLOOR" != "min_one" && "$METHOD" == "aces" ]] && name_suffix="${name_suffix}_${FLOOR}"
[[ -n "$NBINS" && "$METHOD" == "strat" ]] && name_suffix="${name_suffix}_bins${NBINS}"
[[ "$FSCALE" != "10.0" ]] && name_suffix="${name_suffix}_sc${FSCALE}"
[[ "$FMAP"   != "256"  ]] && name_suffix="${name_suffix}_map${FMAP}"
[[ "$HID"    != "155"  ]] && name_suffix="${name_suffix}_hid${HID}"

BOIL=/sdcc/u/gzhao/scratch/inr_sampling/data/PoolBoiling-SubCooled-FC72-2D
NPY=/sdcc/u/gzhao/scratch/inr_sampling/data/NS2d/ns_data_res2048_re10000_7.npy

if [[ "$DATASET" == "boil" ]]; then
  tag=boil
  : "${FRAME:=110}"
  strat_n_bins=8
  data_args=(
    data.dataset_name=PoolBoiling2D data.data_path=$BOIL
    data.poolboiling_condition=100 data.poolboiling_key=temperature
    data.poolboiling_sample_idx=0 data.space_factor=1
  )
elif [[ "$DATASET" == "ns" ]]; then
  tag=ns
  : "${FRAME:=100}"
  strat_n_bins=23
  data_args=(
    data.dataset_name=NS data.data_path=$NPY data.space_factor=1
  )
else
  echo "Unknown DATASET '$DATASET' (expected 'boil' or 'ns')"; exit 1
fi

common=(
  "${data_args[@]}"
  data.seed=$SEED data.data_type=other "data.split_ratios=[1, 0.01, 0.01]"
  data.single_time_frame=$FRAME

  inr.model_type=$MODEL inr.w0=$W0
  inr.depth=3 inr.hidden_dim=$HID
  inr.fourier_mapping_size=$FMAP inr.fourier_scale=$FSCALE
  inr.fourier_activation=relu inr.fourier_include_input=True
  inr.latent_dim=256

  optim.optimizer=sgd optim.epochs=$EPOCHS optim.inner_steps=6 optim.batch_size=2
  optim.sgd_momentum=0.9 optim.sgd_nesterov=True optim.sgd_weight_decay=0.0
  optim.sgd_dampening=0.0 optim.evo_every_epochs=100

  save_sampled_frames=False
  sampling.rate=2e-3
  wandb.use_wandb=True wandb.project=$PROJECT
)

# EVOS reads these; the paper scripts pass them to every run, so they are kept global to
# guarantee the non-EVOS arms are configured identically to the paper's.
evos_common=(
  sampling.sample_num_schedular=constant
  sampling.mutation_method=constant
  sampling.profile_interval_method=lin_dec
  sampling.profile_guide=value
  sampling.n_clusters_2d_start=11
  sampling.n_clusters_2d_end=128
)

case "$METHOD" in
  random)
    method_args=( sampling.type=random )
    ;;
  nmt)
    method_args=( sampling.type=NMT )
    ;;
  evos)
    method_args=( sampling.type=EVOS )
    ;;
  strat)
    method_args=(
      sampling.type=2d_grid_stratified
      sampling.stratified_allocation=neyman
      sampling.stratified_n_bins=${NBINS:-$strat_n_bins}
      sampling.stratified_min_alloc_frac=0.1
      sampling.stratified_update_interval=500
      sampling.stratified_pilot_per_cell=16
    )
    ;;
  aces)
    # The iter=5 / 10%-subdivision variant, NOT the paper-submission `adaptive_best_fast`.
    # Recorded here because the FFN table's ACES row is therefore not the same configuration
    # as the paper's main table.
    method_args=(
      sampling.type=2d_grid_adaptive
      sampling.adaptive_mode=loss_sqrt_std
      sampling.adaptive_iterations=$ACES_ITER
      sampling.subdivision_percentage=$SUBDIV_PCT
      sampling.adaptive_equal_cell_topk=False
      sampling.adaptive_equal_cell_topk_count_mode=same
      sampling.adaptive_equal_cell_topk_weight_mode=none
      sampling.adaptive_weight_mode=loss_powered_weight
      sampling.power_for_loss_as_weight=$POWER
      sampling.adaptive_weight_value_eps=1e-6
      sampling.adaptive_weight_clip_ratio=10
      sampling.adaptive_count_floor_mode=$FLOOR
      sampling.adaptive_grid_update_interval=200
    )
    ;;
  *)
    echo "Unknown METHOD '$METHOD' (expected: random nmt evos strat aces)"; exit 1
    ;;
esac

for lr in $LRS; do
  # ep${EPOCHS} is in the name on purpose: the 100-epoch smoke runs would otherwise be
  # byte-identical in name to the real ones, and extract_wandb_runs.py resolves runs by
  # scanning .wandb files for the name string regardless of which project they went to.
  name="ffn_${tag}_s1_${METHOD}${name_suffix}_lr${lr}_ep${EPOCHS}_t${FRAME}_seed${SEED}"
  echo "=========== ${name} ==========="
  python inr_sample/single_image_inr.py \
      "${common[@]}" "${evos_common[@]}" "${method_args[@]}" \
      optim.lr_inr=$lr \
      wandb.name=$name
  echo "DONE ${name} exit=$?"
done

echo "FFN_S1_${METHOD}_DONE"
