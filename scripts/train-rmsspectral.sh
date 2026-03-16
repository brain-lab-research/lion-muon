#!/bin/bash
#
# Full comparison: RMSspectral variants vs baselines
# Best LRs from tuning (12L, 500 iters, proportional schedule)
#
# Tuning results (500 iters, all peaked):
#   pre        lr=0.03    tune=4.2813
#   pre_ema    lr=0.03    tune=4.2895
#   muon       mlf=0.02   tune=4.6821
#   post_orth  lr=0.001   tune=4.7247
#   adamw      lr=0.001   tune=5.0533
#   split      lr=0.0001  tune=5.3552
#   post       diverges   (all LRs >10.8)
#
# Full comparison results (5000 iters):
#   pre        lr=0.03    val=3.5813  <<<
#   pre_ema    lr=0.03    val=3.5817  <<<
#   adamw      lr=0.001   val=3.6850
#   muon       mlf=0.02   val=3.6955
#   post_orth  lr=0.001   val=4.1115
#   split      lr=0.001   val=6.2149

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
ITERS=64000
WARMUP=3000
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
  --wandb --wandb_project rmsspectral_comparison"

echo "=============================="
echo "  RMSspectral Comparison Run  "
echo "=============================="

# All runs completed. Uncomment to re-run any.

echo "[1/6] Pre (lr=0.03)"
python ./src/main.py $COMMON --opt rmsspectral-pre --lr 0.03 --run_prefix rmsspectral_cmp

echo "[2/6] PreEMA (lr=0.03)"
python ./src/main.py $COMMON --opt rmsspectral-pre_ema --lr 0.03 --run_prefix rmsspectral_cmp

echo "[3/6] AdamW (lr=0.001)"
python ./src/main.py $COMMON --opt adamw --lr 0.001 --run_prefix rmsspectral_cmp

echo "[4/6] Muon (mlf=0.02)"
python ./src/main.py $COMMON --opt muon --lr 0.001 --muon_lr_factor 0.02 --run_prefix rmsspectral_cmp

echo "[5/6] PostOrth (lr=0.001)"
python ./src/main.py $COMMON --opt rmsspectral-post_orth --lr 0.001 --run_prefix rmsspectral_cmp

echo "[6/6] Split (lr=0.0001)"
python ./src/main.py $COMMON --opt rmsspectral-split --lr 0.0001 --run_prefix rmsspectral_cmp

echo "All runs completed. See results in header comments."
