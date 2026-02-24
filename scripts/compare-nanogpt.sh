#!/bin/bash
set -e

cd /home/arman.bolatov/Desktop/llm-baselines

DATASET="fineweb"
MODEL="base"
BATCH_SIZE=32
ACC_STEPS=1
ITERATIONS=10000
EVAL_INTERVAL=1000
SEQUENCE_LENGTH=512
DEVICE="cuda:6"

# NanoGPT configuration (base model)
N_LAYER=12
N_HEAD=12
N_EMBD=768

# Shared training hyperparameters
LR=3e-4
MOMENTUM=0.95
NS_STEPS=6
WEIGHT_DECAY=0.1
BETA1=0.9
BETA2=0.95

COMMON_ARGS="--dataset $DATASET \
  --model $MODEL \
  --batch_size $BATCH_SIZE \
  --acc_steps $ACC_STEPS \
  --iterations $ITERATIONS \
  --eval_interval $EVAL_INTERVAL \
  --sequence_length $SEQUENCE_LENGTH \
  --n_layer $N_LAYER \
  --n_head $N_HEAD \
  --n_embd $N_EMBD \
  --device $DEVICE \
  --scheduler wsd \
  --warmup_steps 1000 \
  --wandb \
  --wandb_project llm-baselines-compare"

echo "=========================================="
echo "Compare: RMSspectral-SANIA | AdaMuon | RMSspectral | Muon | AdamW"
echo "Model: NanoGPT (base) | Dataset: $DATASET | Steps: $ITERATIONS | LR: $LR"
echo "=========================================="

# 1) RMSspectral-SANIA
echo "\n[1/5] Running RMSspectral-SANIA..."
python ./src/main.py \
  $COMMON_ARGS \
  --opt rmsspectral-sania \
  --lr $LR \
  --momentum $MOMENTUM \
  --adamuon_ns_steps $NS_STEPS \
  --weight_decay $WEIGHT_DECAY \
  --wandb_run_prefix compare_rmsspectral_sania_nanogpt

# 2) RMSspectral
echo "\n[2/5] Running RMSspectral..."
python ./src/main.py \
  $COMMON_ARGS \
  --opt rmsspectral \
  --lr $LR \
  --momentum $MOMENTUM \
  --adamuon_ns_steps $NS_STEPS \
  --weight_decay $WEIGHT_DECAY \
  --wandb_run_prefix compare_rmsspectral_nanogpt

# 3) AdaMuon
echo "\n[3/5] Running AdaMuon..."
python ./src/main.py \
  $COMMON_ARGS \
  --opt adamuon \
  --lr $LR \
  --momentum $MOMENTUM \
  --adamuon_ns_steps $NS_STEPS \
  --weight_decay $WEIGHT_DECAY \
  --adamuon_rms_factor 0.2 \
  --wandb_run_prefix compare_adamuon_nanogpt

# 4) Muon
echo "\n[4/5] Running Muon..."
python ./src/main.py \
  $COMMON_ARGS \
  --opt muon \
  --lr $LR \
  --muon_lr_factor 0.02 \
  --muon_ns_steps $NS_STEPS \
  --momentum $MOMENTUM \
  --nesterov True \
  --weight_decay $WEIGHT_DECAY \
  --beta1 $BETA1 \
  --beta2 $BETA2 \
  --wandb_run_prefix compare_muon_nanogpt

# 5) AdamW
echo "\n[5/5] Running AdamW..."
python ./src/main.py \
  $COMMON_ARGS \
  --opt adamw \
  --lr $LR \
  --beta1 $BETA1 \
  --beta2 $BETA2 \
  --weight_decay $WEIGHT_DECAY \
  --wandb_run_prefix compare_adamw_nanogpt

echo "\nAll comparisons completed. Check WandB project 'llm-baselines-compare'."
