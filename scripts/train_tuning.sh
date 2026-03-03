#!/bin/bash
set -e

export OMP_NUM_THREADS=1

DEVICE="cuda:0"
ITERATIONS=2000
WARMUP=200
EVAL_INTERVAL=100
MAX_JOBS=1  # 12L/768d is large

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
  --wandb \
  --wandb_project sign-muon-tuning"

SHARED="--beta1 0.9 --beta2 0.95 --momentum 0.95 --nesterov True \
  --muon_ns_steps 6 --weight_decay 0.1"

PIDS=()

cleanup() {
  echo -e "\nCaught interrupt, killing all jobs..."
  kill "${PIDS[@]}" 2>/dev/null
  wait 2>/dev/null
  exit 1
}
trap cleanup INT TERM

# Remove finished PIDs from the array
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
  # Block until a slot opens
  reap
  while [ ${#PIDS[@]} -ge $MAX_JOBS ]; do
    sleep 1
    reap
  done
  echo "[RUN]  $name"
  OMP_NUM_THREADS=1 python ./src/main.py $COMMON_ARGS $SHARED \
    --experiment_name "$name" "$@" &
  PIDS+=($!)
}

# ── 1. AdamW: sweep first (likely undertuned) ──
for LR in 2e-3 1e-3 5e-3 5e-4 1e-2 2e-4 1e-4; do
  run "adam_lr${LR}" \
    --opt adamw --lr $LR
done

# ── 2. Muon LR sweep (0.01 was at edge, extend upward) ──
for LR in 1e-2 5e-3 2e-2 2e-3 4e-2 1e-3 5e-4 2e-4 1e-4; do
  run "muon_lr${LR}" \
    --opt sign_muon --lr $LR --muon_lr_factor $LR \
    --muon_every_k 1 --cheap_mode sign
done

# ── 3. Lion LR sweep ──
for LR in 2e-4 1e-4 5e-4 5e-5 1e-3 2e-5; do
  run "lion_lr${LR}" \
    --opt lion --lr $LR --beta1 0.9 --beta2 0.99
done

# ── 4. Signum LR sweep ──
for LR in 5e-4 1e-3 2e-3 2e-4 5e-3 1e-4 1e-2; do
  run "signum_lr${LR}" \
    --opt signum --lr $LR
done

# ── 5. SignMuon K=5: tune sign_lr ──
for LR in 2e-3 5e-3 1e-3; do
  for SLR in 5e-4 1e-4 1e-3 5e-3 5e-5 2e-5; do
    run "signmuon_lr${LR}_slr${SLR}" \
      --opt sign_muon --lr $LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 5 --cheap_mode sign \
      --sign_scaling none
  done
done

# ── 6. LionMuon K=5: tune lr and sign_lr (uses Lion betas, not SHARED) ──
for LR in 2e-3 5e-3 1e-3; do
  for SLR in 5e-5 1e-4 5e-4 1e-3 2e-5 1e-5; do
    run "lionmuon_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 5 \
      --beta1 0.95 --beta2 0.99
  done
done

# ── 7. SignMuon K=20: tune lr and sign_lr ──
for LR in 2e-3 5e-3 1e-3; do
  for SLR in 5e-5 1e-4 2e-5 5e-4; do
    run "signmuon_k20_lr${LR}_slr${SLR}" \
      --opt sign_muon --lr $LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 20 --cheap_mode sign \
      --sign_scaling none
  done
done

# ── 8. LionMuon K=20: tune lr and sign_lr ──
for LR in 2e-3 5e-3 1e-3; do
  for SLR in 5e-5 1e-4 2e-5 5e-4; do
    run "lionmuon_k20_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 20 \
      --beta1 0.95 --beta2 0.99
  done
done

# ── 9. SignMuon K=2: tune lr and sign_lr ──
for LR in 2e-3 5e-3 1e-3; do
  for SLR in 5e-5 1e-4 2e-5 5e-4; do
    run "signmuon_k2_lr${LR}_slr${SLR}" \
      --opt sign_muon --lr $LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 2 --cheap_mode sign \
      --sign_scaling none
  done
done

# ── 10. LionMuon K=2: tune lr and sign_lr ──
for LR in 2e-3 5e-3 1e-3; do
  for SLR in 5e-5 1e-4 2e-5 5e-4; do
    run "lionmuon_k2_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 2 \
      --beta1 0.95 --beta2 0.99
  done
done

# ── 11. LionMuon K=1: Muon with Lion dual EMA, tune lr only ──
for LR in 1e-2 5e-3 2e-2 2e-3 4e-2 1e-3; do
  run "lionmuon_k1_lr${LR}" \
    --opt lion_muon --lr $LR --muon_lr_factor $LR \
    --muon_every_k 1 \
    --beta1 0.95 --beta2 0.99
done

# Wait for all remaining jobs
echo "Waiting for remaining jobs..."
wait
echo "Done!"
