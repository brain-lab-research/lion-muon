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
DEVICE="cuda:7"

# NanoGPT configuration (base model)
N_LAYER=12
N_HEAD=12
N_EMBD=768

# Shared training hyperparameters
LR=3e-3
BETA1=0.95
BETA2=0.95
WEIGHT_DECAY=0.01
PRECONDITION_FREQUENCY=10
MAX_PRECOND_DIM=10000
MERGE_DIMS=false
PRECONDITION_1D=false
NORMALIZE_GRADS=false
SOAP_DATA_FORMAT="channels_first"
CORRECT_BIAS=true

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

echo "\n[Run] SOAP-Sania..."
python ./src/main.py \
  $COMMON_ARGS \
  --opt soap-sania \
  --lr $LR \
  --beta1 $BETA1 \
  --beta2 $BETA2 \
  --weight_decay $WEIGHT_DECAY \
  --precondition_frequency $PRECONDITION_FREQUENCY \
  --max_precond_dim $MAX_PRECOND_DIM \
  --merge_dims $MERGE_DIMS \
  --precondition_1d $PRECONDITION_1D \
  --normalize_grads $NORMALIZE_GRADS \
  --soap_data_format $SOAP_DATA_FORMAT \
  --correct_bias $CORRECT_BIAS \
  --wandb_run_prefix soap_sania
