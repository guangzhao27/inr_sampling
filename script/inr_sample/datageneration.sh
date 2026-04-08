#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m2956_g
#SBATCH --array=0-1

cd /pscratch/sd/g/gzhao27/INR/INR_SAMPLE
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torchgeo

# Use array mode by default: one viscosity per job for easier 1024x1024 runs.
NU_VALUES=(0.001 0.05)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NU="${NU_VALUES[$TASK_ID]}"

# Override any setting at submit time, e.g.:
#   sbatch --export=ALL,N_TRAIN_NU=20,DT=0.0008,SUBSTEPS=80 script/inr_sample/datageneration.sh
OUT_DIR="${OUT_DIR:-/pscratch/sd/g/gzhao27/INR/data/2D_Burgers_1024}"
H="${H:-1024}"
W="${W:-1024}"
T_STEPS="${T_STEPS:-100}"
DT="${DT:-0.001}"
SUBSTEPS="${SUBSTEPS:-100}"
N_TRAIN_NU="${N_TRAIN_NU:-10}"

python /pscratch/sd/g/gzhao27/INR/INR_SAMPLE/utils/data/generate_burgers2d.py \
  --H "$H" --W "$W" --T_steps "$T_STEPS" \
  --dt "$DT" --substeps "$SUBSTEPS" \
  --N_train_nu "$N_TRAIN_NU" \
  --nu_list "$NU" \
  --resume --flush_every 1 \
  --max_retries 6 --dt_shrink 0.5 --min_dt 1e-5 --rms_shrink 0.8 \
  --out_dir "$OUT_DIR"