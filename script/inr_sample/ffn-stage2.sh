#!/bin/bash
#SBATCH -p csi
#SBATCH -t 04:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1
#SBATCH --job-name=ffn-s2
#SBATCH --output=slurm-ffn-s2-%j.out

# Stage 2 of the FFN architecture-generality experiment: the frame x seed matrix that turns
# Stage 1's single number into a paired statistical comparison.
#
#   5 methods x 3 frames x seeds {42, 43, 44} = 45 cells
#   boil: frames 30/110/190, tuning cell t110   |   ns: frames 100/140/180, tuning cell t100
#
# Each method runs at its OWN Stage-1 best lr. Sampler configuration is not duplicated here:
# this driver delegates to ffn-stage1-lr.sh with LRS pinned to one value, so there is exactly
# one definition of what "ACES" or "stratified" means. Partition knobs (ACES_ITER,
# SUBDIV_PCT, FLOOR, NBINS) pass straight through the environment.
#
# The Stage-1 tuning cell at each method's best lr was already produced by Stage 1 and is
# deliberately NOT re-run: the config is identical, and a second run would write a duplicate
# W&B run name, which the name-scanning extractor cannot disambiguate.
#
#   for m in random nmt evos strat aces; do
#     sbatch --export=ALL,DATASET=boil,METHOD=$m script/inr_sample/ffn-stage2.sh
#   done
#
# Corrected NS partitions (cell-count sweep, Agent-report/ffn/ffn-ns1024.md section 9):
#   ACES  iter=8 pct=20 FLOOR=soft -> 2665 leaves  (vs 226 at the pinned iter=5)
#   strat NBINS=32 -> 1024 cells                   (vs 529 from the n/4 rule of thumb)
#   sbatch --export=ALL,DATASET=ns,METHOD=aces,ACES_ITER=8,SUBDIV_PCT=20,FLOOR=soft script/inr_sample/ffn-stage2.sh
#   sbatch --export=ALL,DATASET=ns,METHOD=strat,NBINS=32 script/inr_sample/ffn-stage2.sh

REPO_ROOT="/hpcgpfs01/scratch/gzhao/inr_sampling"
cd "$REPO_ROOT"

: "${DATASET:=boil}"
: "${METHOD:=random}"

# Best lr per (dataset, method) from the Stage-1 sweeps. Literals, not read from JSON, so a
# Stage-2 job is reproducible from this file alone.
if [[ "$DATASET" == "boil" ]]; then
  frames=(30 110 190); tune_frame=110
  case "$METHOD" in
    random) BEST_LR=1e-1 ;;
    nmt)    BEST_LR=1e-2 ;;
    evos)   BEST_LR=3e-2 ;;
    strat)  BEST_LR=1e-1 ;;
    aces)   BEST_LR=1e-1 ;;
    *) echo "Unknown METHOD '$METHOD'"; exit 1 ;;
  esac
elif [[ "$DATASET" == "ns" ]]; then
  frames=(100 140 180); tune_frame=100
  case "$METHOD" in
    random) BEST_LR=3e-2 ;;
    nmt)    BEST_LR=3e-3 ;;
    evos)   BEST_LR=1e-2 ;;
    strat)  BEST_LR=3e-2 ;;
    aces)   BEST_LR=1e-2 ;;
    *) echo "Unknown METHOD '$METHOD'"; exit 1 ;;
  esac
else
  echo "Unknown DATASET '$DATASET'"; exit 1
fi

echo "STAGE2 dataset=$DATASET method=$METHOD best_lr=$BEST_LR"
echo "       ACES_ITER=${ACES_ITER:-5} SUBDIV_PCT=${SUBDIV_PCT:-10} FLOOR=${FLOOR:-min_one} NBINS=${NBINS:-default}"

for t in "${frames[@]}"; do
  for s in 42 43 44; do
    if [[ "$t" == "$tune_frame" && "$s" == "42" ]]; then
      echo "SKIP t=$t seed=42 (already run in Stage 1 at lr=$BEST_LR)"
      continue
    fi
    DATASET=$DATASET METHOD=$METHOD FRAME=$t SEED=$s LRS="$BEST_LR" \
      bash script/inr_sample/ffn-stage1-lr.sh
  done
done

echo "FFN_S2_${METHOD}_DONE"
