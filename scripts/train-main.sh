#!/bin/bash
set -e

export OMP_NUM_THREADS=1

cd /home/arman.bolatov/Desktop/llm-baselines

DEVICE="cuda:5"
ITERATIONS=10000
WARMUP=1000
EVAL_INTERVAL=200

COMMON_ARGS="--dataset fineweb \
  --model base \
  --batch_size 32 \
  --acc_steps 1 \
  --iterations $ITERATIONS \
  --eval_interval $EVAL_INTERVAL \
  --sequence_length 512 \
  --n_layer 12 \
  --n_head 12 \
  --n_embd 768 \
  --device $DEVICE \
  --scheduler wsd \
  --warmup_steps $WARMUP \
  --weight_decay 0.1 \
  --muon_ns_steps 6 \
  --wandb \
  --wandb_project sign-muon-main"

run() {
  local name=$1; shift
  if [ -f "exps/${name}/summary.json" ]; then
    echo "[SKIP] $name"
    return
  fi
  echo "[RUN]  $name"
  python ./src/main.py $COMMON_ARGS --experiment_name "$name" "$@"
}

# Best hyperparams from sign-muon-tuning-v2 (6L/384d, 3K iters):

run "adamw" \
  --opt adamw --lr 2e-3 --beta1 0.9 --beta2 0.95

run "signum" \
  --opt signum --lr 5e-4 --momentum 0.9

run "muon" \
  --opt sign_muon --lr 2e-3 --muon_lr_factor 2e-3 \
  --muon_every_k 1 --cheap_mode sign \
  --momentum 0.95 --nesterov True --beta1 0.9 --beta2 0.95

run "lion" \
  --opt lion --lr 2e-4 --beta1 0.9 --beta2 0.99

run "signmuon_k5" \
  --opt sign_muon --lr 5e-3 --muon_lr_factor 5e-3 \
  --sign_lr 1e-4 --muon_every_k 5 --cheap_mode sign --sign_scaling none \
  --momentum 0.95 --nesterov True --beta1 0.9 --beta2 0.95

run "lionmuon_k5" \
  --opt lion_muon --lr 2e-3 --muon_lr_factor 2e-3 \
  --sign_lr 5e-5 --muon_every_k 5 \
  --beta1 0.95 --beta2 0.99

echo "Done!"
