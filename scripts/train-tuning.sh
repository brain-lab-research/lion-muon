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
  --wandb_project sign-muon-tuning"

SHARED="--beta1 0.9 --beta2 0.95 --momentum 0.95 --nesterov True \
  --muon_ns_steps 6 --weight_decay 0.1"

run() {
  local name=$1; shift
  if [ -f "exps/${name}/summary.json" ]; then
    echo "[SKIP] $name"
    return
  fi
  echo "[RUN]  $name"
  python ./src/main.py $COMMON_ARGS $SHARED --experiment_name "$name" "$@"
}

# 1. Sweep base LR: Adam and Muon share the same LR (Moonshot scaling baked in)
for LR in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2; do
  run "adam_lr${LR}" \
    --opt adamw --lr $LR

  run "muon_lr${LR}" \
    --opt sign_muon --lr $LR --muon_lr_factor $LR \
    --muon_every_k 1 --cheap_mode sign
done

# 2. Signum LR sweep
for LR in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2; do
  run "signum_lr${LR}" \
    --opt signum --lr $LR
done

# 3. SignMuon K=5: tune sign_lr only
#    Base LR same as Adam/Muon, sign uses raw sign with its own LR
for LR in 1e-3 2e-3 5e-3; do
  for SLR in 1e-4 5e-4 1e-3 5e-3; do
    run "signmuon_lr${LR}_slr${SLR}" \
      --opt sign_muon --lr $LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 5 --cheap_mode sign \
      --sign_scaling none
  done
done

echo "Done!"
