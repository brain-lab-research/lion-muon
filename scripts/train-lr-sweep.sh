#!/bin/bash
set -e

export OMP_NUM_THREADS=1

cd /home/arman.bolatov/Desktop/llm-baselines

DEVICE="cuda:5"
ITERATIONS=3000
WARMUP=300
EVAL_INTERVAL=100

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
  --wandb_project sign-muon-lr-sweep"

SHARED="--beta1 0.9 --beta2 0.95 --momentum 0.95 --nesterov True \
  --muon_ns_steps 6 --weight_decay 0.1"

run() {
  local name=$1; shift
  echo "[RUN]  $name"
  python ./src/main.py $COMMON_ARGS $SHARED --experiment_name "$name" "$@"
}

# Muon baseline: sweep muon_lr
for LR in 0.01 0.02 0.04 0.08; do
  run "muon_lr${LR}" \
    --opt muon --lr 3e-4 --muon_lr_factor $LR
done

# Signum baseline: sweep lr
for LR in 0.001 0.002 0.005 0.01; do
  run "signum_lr${LR}" \
    --opt signum --lr $LR
done

# sign_k2: sweep muon_lr, sign_lr auto-scales to 0.25x
for LR in 0.01 0.02 0.04 0.08; do
  run "sign_k2_lr${LR}" \
    --opt sign_muon --lr 3e-4 --muon_lr_factor $LR \
    --muon_every_k 2 --cheap_mode sign
done

echo "Done!"
