#!/bin/bash
set -e

export OMP_NUM_THREADS=1

cd /home/arman.bolatov/Desktop/llm-baselines

DEVICE="cuda:5"
ITERATIONS=3000
WARMUP=300
EVAL_INTERVAL=100
K=5

COMMON_ARGS="--dataset fineweb \
  --model base \
  --batch_size 32 \
  --acc_steps 1 \
  --iterations $ITERATIONS \
  --eval_interval $EVAL_INTERVAL \
  --sequence_length 256 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --device $DEVICE \
  --scheduler wsd \
  --warmup_steps $WARMUP \
  --wandb \
  --wandb_project sign-muon-lr-ablation-v2"

MUON_LR=0.02
ADAMW_LR=3e-4
SHARED="--beta1 0.9 --beta2 0.95 --momentum 0.95 --nesterov True \
  --muon_ns_steps 6 --weight_decay 0.1"

is_completed() {
  [ -f "exps/${1}/summary.json" ] && return 0
  return 1
}

for SIGN_LR in 0.001 0.002 0.005 0.01; do
  name="sign_muon_k${K}_slr${SIGN_LR}"
  if is_completed "$name"; then
    echo "[SKIP] $name"
    continue
  fi
  echo "[RUN]  $name"
  python ./src/main.py $COMMON_ARGS $SHARED \
    --opt sign_muon --lr $ADAMW_LR --muon_lr_factor $MUON_LR \
    --sign_lr $SIGN_LR --muon_every_k $K \
    --experiment_name "$name"
done

echo "Done!"
