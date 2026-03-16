#!/bin/bash
#
# LR sweep for RMSspectral variants + baselines
# Usage: bash scripts/tune-rmsspectral.sh <gpu>
#

set -e

GPU=${1:-0}
echo "Using GPU: $GPU"

cd "$(dirname "$0")/.."

MODEL="llama"
N_EMBD=768
N_HEAD=12
N_LAYER=12
SEQ_LEN=512
BATCH=50
ACC=4
ITERS=500
WARMUP=50
SCHEDULER="cos"
DATASET="fineweb"
WD=0.1
BETA1=0.9
BETA2=0.95
MOMENTUM=0.95
NS_STEPS=5

COMMON="--model $MODEL --n_embd $N_EMBD --n_head $N_HEAD --n_layer $N_LAYER \
  --sequence_length $SEQ_LEN --batch_size $BATCH --acc_steps $ACC \
  --iterations $ITERS --warmup_steps $WARMUP --scheduler $SCHEDULER \
  --dataset $DATASET --weight_decay $WD --beta1 $BETA1 --beta2 $BETA2 \
  --momentum $MOMENTUM --muon_ns_steps $NS_STEPS --nesterov True \
  --device cuda:$GPU \
  --wandb --wandb_project rmsspectral_tune"

# Per-variant LR ranges (includes boundary extensions)
LRS_PRE="0.0003 0.001 0.01 0.03 0.05 0.1"
LRS_PRE_EMA="0.0003 0.001 0.01 0.03 0.05 0.1"
LRS_POST_ORTH="0.0001 0.0003 0.001 0.003 0.005"
LRS_SPLIT="0.00001 0.00003 0.0001 0.0003 0.001"
LRS_ADAMW="0.0003 0.001 0.003 0.01 0.03"
MUON_LR_FACTORS="0.005 0.01 0.02 0.05"

# Skip if already completed
is_complete() {
  local opt="$1" key="$2"
  local prefix="tune_lr_${BATCH}x${ACC}_model-${MODEL}_dataset-${DATASET}_opt-${opt}"
  for dir in ./exps/${prefix}*; do
    [[ -f "${dir}/summary.json" ]] && [[ "$dir" == *"${key}"* ]] && return 0
  done
  return 1
}

echo "=============================="
echo "  RMSspectral LR Sweep        "
echo "=============================="

for LR in $LRS_PRE; do
  if is_complete "rmsspectral-pre" "lr-${LR}"; then
    echo "SKIP (done): rmsspectral-pre lr=$LR"; continue
  fi
  echo ">>> rmsspectral-pre lr=$LR"
  python ./src/main.py $COMMON --opt rmsspectral-pre --lr $LR --run_prefix tune_lr
done

for LR in $LRS_PRE_EMA; do
  if is_complete "rmsspectral-pre_ema" "lr-${LR}"; then
    echo "SKIP (done): rmsspectral-pre_ema lr=$LR"; continue
  fi
  echo ">>> rmsspectral-pre_ema lr=$LR"
  python ./src/main.py $COMMON --opt rmsspectral-pre_ema --lr $LR --run_prefix tune_lr
done

for LR in $LRS_POST_ORTH; do
  if is_complete "rmsspectral-post_orth" "lr-${LR}"; then
    echo "SKIP (done): rmsspectral-post_orth lr=$LR"; continue
  fi
  echo ">>> rmsspectral-post_orth lr=$LR"
  python ./src/main.py $COMMON --opt rmsspectral-post_orth --lr $LR --run_prefix tune_lr
done

for LR in $LRS_SPLIT; do
  if is_complete "rmsspectral-split" "lr-${LR}"; then
    echo "SKIP (done): rmsspectral-split lr=$LR"; continue
  fi
  echo ">>> rmsspectral-split lr=$LR"
  python ./src/main.py $COMMON --opt rmsspectral-split --lr $LR --run_prefix tune_lr
done

for LR in $LRS_ADAMW; do
  if is_complete "adamw" "lr-${LR}"; then
    echo "SKIP (done): adamw lr=$LR"; continue
  fi
  echo ">>> AdamW lr=$LR"
  python ./src/main.py $COMMON --opt adamw --lr $LR --run_prefix tune_lr
done

for F in $MUON_LR_FACTORS; do
  if is_complete "muon" "muon_lr_factor-${F}"; then
    echo "SKIP (done): muon mlf=$F"; continue
  fi
  echo ">>> Muon muon_lr_factor=$F"
  python ./src/main.py $COMMON --opt muon --lr 0.001 --muon_lr_factor $F --run_prefix tune_lr
done

echo "Done! Check wandb project: rmsspectral_tune"
