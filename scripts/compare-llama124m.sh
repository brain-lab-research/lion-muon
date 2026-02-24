#!/bin/bash
set -e

cd /home/arman.bolatov/Desktop/llm-baselines

DATASET="fineweb"
MODEL="llama"
BATCH_SIZE=16
ACC_STEPS=1
ITERATIONS=10000
EVAL_INTERVAL=1000
SEQUENCE_LENGTH=256
DEVICE="cuda:7"

# Llama 124M configuration
N_LAYER=12
N_HEAD=12
N_EMBD=768
N_KV_HEAD=12

# Shared training hyperparameters
LR=5e-2
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
  --n_kv_head $N_KV_HEAD \
  --device $DEVICE \
  --scheduler wsd \
  --warmup_steps 1000 \
  --wandb \
  --wandb_project llm-baselines-llama"

echo "=========================================="
echo "Compare: rmsspectral-sania | adamuon | rmsspectral | muon | adamw"
echo "Model: Llama 124M | Dataset: $DATASET | Steps: $ITERATIONS | LR: $LR | WD: $WEIGHT_DECAY"
echo "=========================================="

for LR in 5e-4 3e-4 1e-3; do
  # echo "\n===== Starting comparisons with LR=$LR ====="
  # echo "\n[1/5] Running RMSSpectral-SANIA (p=0.5)..."
  # python ./src/main.py \
  #   $COMMON_ARGS \
  #   --opt rmsspectral-sania \
  #   --lr $LR \
  #   --beta1 $BETA1 \
  #   --beta2 $BETA2 \
  #   --muon_ns_steps $NS_STEPS \
  #   --weight_decay $WEIGHT_DECAY \
  #   --wandb_run_prefix compare_rmsspectral_sania_llama

  # echo "\n[2/5] Running AdaMuon..."
  # python ./src/main.py \
  #   $COMMON_ARGS \
  #   --opt adamuon \
  #   --lr $LR \
  #   --momentum $MOMENTUM \
  #   --muon_ns_steps $NS_STEPS \
  #   --weight_decay $WEIGHT_DECAY \
  #   --beta1 $BETA1 \
  #   --beta2 $BETA2 \
  #   --wandb_run_prefix compare_adamuon_llama

  # echo "\n[3/5] Running RMSSpectral (p=0.25)..."
  # python ./src/main.py \
  #   $COMMON_ARGS \
  #   --opt rmsspectral \
  #   --lr $LR \
  #   --beta1 $BETA1 \
  #   --beta2 $BETA2 \
  #   --muon_ns_steps $NS_STEPS \
  #   --weight_decay $WEIGHT_DECAY \
  #   --wandb_run_prefix compare_rmsspectral_llama

  # echo "\n[4/5] Running Muon..."
  # python ./src/main.py \
  #   $COMMON_ARGS \
  #   --opt muon \
  #   --lr $LR \
  #   --muon_lr_factor 0.02 \
  #   --muon_ns_steps $NS_STEPS \
  #   --momentum $MOMENTUM \
  #   --nesterov True \
  #   --weight_decay $WEIGHT_DECAY \
  #   --beta1 $BETA1 \
  #   --beta2 $BETA2 \
  #   --wandb_run_prefix compare_muon_llama

  # echo "\n[5/6] Running Adam-SANIA..."
  # python ./src/main.py \
  #   $COMMON_ARGS \
  #   --opt adam-sania \
  #   --lr $LR \
  #   --beta1 $BETA1 \
  #   --beta2 $BETA2 \
  #   --weight_decay $WEIGHT_DECAY \
  #   --wandb_run_prefix compare_adam_sania_llama

  # echo "\n[6/7] Running SOAP-Sania..."
  # python ./src/main.py \
  #   $COMMON_ARGS \
  #   --opt soap-sania \
  #   --lr $LR \
  #   --beta1 $BETA1 \
  #   --beta2 $BETA2 \
  #   --weight_decay $WEIGHT_DECAY \
  #   --wandb_run_prefix compare_soap_sania_llama

  echo "\n[6/7] Running Shampoo..."
  python ./src/main.py \
    $COMMON_ARGS \
    --opt shampoo \
    --lr $LR \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --weight_decay $WEIGHT_DECAY \
    --wandb_run_prefix compare_shampoo_llama

  echo "\n[7/7] Running Shampoo-Sania..."
  python ./src/main.py \
    $COMMON_ARGS \
    --opt shampoo-sania \
    --lr $LR \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --weight_decay $WEIGHT_DECAY \
    --wandb_run_prefix compare_shampoo_sania_llama

  # echo "\n[5/5] Running AdamW..."
  # python ./src/main.py \
  #   $COMMON_ARGS \
  #   --opt adamw \
  #   --lr $LR \
  #   --beta1 $BETA1 \
  #   --beta2 $BETA2 \
  #   --weight_decay $WEIGHT_DECAY \
  #   --wandb_run_prefix compare_adamw_llama

echo "\nAll comparisons completed. Check WandB project 'llm-baselines-compare'."
done;
