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
  --weight_decay 0.1 \
  --muon_ns_steps 6 \
  --wandb \
  --wandb_project sign-muon-tuning-v2"

run() {
  local name=$1; shift
  if [ -f "exps/${name}/summary.json" ]; then
    echo "[SKIP] $name"
    return
  fi
  echo "[RUN]  $name"
  python ./src/main.py $COMMON_ARGS --experiment_name "$name" "$@"
}

# ── AdamW: sweep lr ────────────────────────────────────────────────────────
for LR in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3; do
  run "adam_lr${LR}" \
    --opt adamw --lr $LR --beta1 0.9 --beta2 0.95
done

# ── Signum: sweep lr ───────────────────────────────────────────────────────
for LR in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3; do
  run "signum_lr${LR}" \
    --opt signum --lr $LR --momentum 0.9
done

# ── Muon (K=1, Moonshot scaling): sweep lr ────────────────────────────────
for LR in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3; do
  run "muon_lr${LR}" \
    --opt sign_muon --lr $LR --muon_lr_factor $LR \
    --muon_every_k 1 --cheap_mode sign \
    --momentum 0.95 --nesterov True \
    --beta1 0.9 --beta2 0.95
done

# ── Lion: sweep lr (beta1=0.9 per paper, extended range downward) ──────────
for LR in 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3; do
  run "lion_lr${LR}" \
    --opt lion --lr $LR --beta1 0.9 --beta2 0.99
done

# ── SignMuon K=5: 3 base lrs × 4 sign_lrs ─────────────────────────────────
for LR in 1e-3 2e-3 5e-3; do
  for SLR in 1e-4 5e-4 1e-3 5e-3; do
    run "signmuon_lr${LR}_slr${SLR}" \
      --opt sign_muon --lr $LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 5 --cheap_mode sign \
      --sign_scaling none \
      --momentum 0.95 --nesterov True \
      --beta1 0.9 --beta2 0.95
  done
done

# ── LionMuon K=5: 3 base lrs × 4 lion_lr ratios ───────────────────────────
# lion_lr is passed as sign_lr; stored as ratio (lion_lr/muon_lr), scales with scheduler
for LR in 1e-3 2e-3 5e-3; do
  for LLR in 1e-5 5e-5 1e-4 5e-4; do
    run "lionmuon_lr${LR}_llr${LLR}" \
      --opt lion_muon --lr $LR --muon_lr_factor $LR \
      --sign_lr $LLR --muon_every_k 5 \
      --beta1 0.95 --beta2 0.99
  done
done

echo "Done!"
