#!/bin/bash
set -e

export OMP_NUM_THREADS=1

cd /home/arman/llm-baselines/

DEVICE="cuda:0"
ITERATIONS=20000
WARMUP=2000
EVAL_INTERVAL=200
MAX_JOBS=1  # 12L/768d is bigger, 2 should fit

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
  --scheduler cos \
  --warmup_steps $WARMUP \
  --weight_decay 0.1 \
  --muon_ns_steps 6 \
  --wandb \
  --wandb_project sign-muon-main"

PIDS=()

cleanup() {
  echo -e "\nCaught interrupt, killing all jobs..."
  kill "${PIDS[@]}" 2>/dev/null
  wait 2>/dev/null
  exit 1
}
trap cleanup INT TERM

reap() {
  local alive=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  PIDS=("${alive[@]}")
}

run() {
  local name=$1; shift
  if [ -f "exps/${name}/summary.json" ]; then
    echo "[SKIP] $name"
    return
  fi
  reap
  while [ ${#PIDS[@]} -ge $MAX_JOBS ]; do
    sleep 1
    reap
  done
  echo "[RUN]  $name"
  OMP_NUM_THREADS=1 python ./src/main.py $COMMON_ARGS \
    --experiment_name "$name" "$@" &
  PIDS+=($!)
}

# Best hyperparams from sign-muon-tuning (6L/384d, 2K iters, cos sched):

run "adamw" \
  --opt adamw --lr 1e-3 --beta1 0.9 --beta2 0.95

run "lion" \
  --opt lion --lr 1e-4 --beta1 0.9 --beta2 0.99

run "signum" \
  --opt signum --lr 2e-4 --momentum 0.9

run "muon" \
  --opt sign_muon --lr 5e-3 --muon_lr_factor 5e-3 \
  --muon_every_k 1 --cheap_mode sign \
  --momentum 0.95 --nesterov True --beta1 0.9 --beta2 0.95

run "signmuon_k5" \
  --opt sign_muon --lr 5e-3 --muon_lr_factor 5e-3 \
  --sign_lr 5e-5 --muon_every_k 5 --cheap_mode sign --sign_scaling none \
  --momentum 0.95 --nesterov True --beta1 0.9 --beta2 0.95

run "lionmuon_k5" \
  --opt lion_muon --lr 5e-3 --muon_lr_factor 5e-3 \
  --sign_lr 5e-5 --muon_every_k 5 \
  --beta1 0.95 --beta2 0.99

run "signmuon_k2" \
  --opt sign_muon --lr 5e-3 --muon_lr_factor 5e-3 \
  --sign_lr 5e-5 --muon_every_k 2 --cheap_mode sign --sign_scaling none \
  --momentum 0.95 --nesterov True --beta1 0.9 --beta2 0.95

run "lionmuon_k2" \
  --opt lion_muon --lr 5e-3 --muon_lr_factor 5e-3 \
  --sign_lr 5e-5 --muon_every_k 2 \
  --beta1 0.95 --beta2 0.99

# Note: K=2 tuning confirmed lr=5e-3 slr=5e-5 as best for both (same as K=5,K=20)

run "signmuon_k20" \
  --opt sign_muon --lr 5e-3 --muon_lr_factor 5e-3 \
  --sign_lr 5e-5 --muon_every_k 20 --cheap_mode sign --sign_scaling none \
  --momentum 0.95 --nesterov True --beta1 0.9 --beta2 0.95

run "lionmuon_k20" \
  --opt lion_muon --lr 5e-3 --muon_lr_factor 5e-3 \
  --sign_lr 5e-5 --muon_every_k 20 \
  --beta1 0.95 --beta2 0.99

# LionMuon K=1: every step is Muon but with Lion dual EMA (β1=0.95,β2=0.99)
# This is NOT the same as "Muon" which uses SGD momentum (μ=0.95)
run "lionmuon_k1" \
  --opt lion_muon --lr 2e-3 --muon_lr_factor 2e-3 \
  --muon_every_k 1 \
  --beta1 0.95 --beta2 0.99

echo "Waiting for remaining jobs..."
wait
echo "Done!"
