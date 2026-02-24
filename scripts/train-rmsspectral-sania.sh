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

echo "\n[Run] RMSSpectral-SANIA (p=0.5)..."
python ./src/main.py \
  $COMMON_ARGS \
  --opt rmsspectral-sania \
  --lr $LR \
  --beta1 $BETA1 \
  --beta2 $BETA2 \
  --muon_ns_steps $NS_STEPS \
  --momentum $MOMENTUM \
  --weight_decay $WEIGHT_DECAY \
  --wandb_run_prefix rmsspectral-sania
