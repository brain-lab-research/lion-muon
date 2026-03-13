#!/bin/bash
set -e

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/common_config.sh"

PYTHON=/data/users/arman/miniconda3/envs/optim/bin/python

GPU_ID="${1:-0}"
DEVICE="cuda:${GPU_ID}"

ITERATIONS=2000
WARMUP=200
EVAL_INTERVAL=100
MAX_JOBS=2  # run at most 2 experiments in parallel

EXPS_DIR=./exps_tuning_gpt

COMMON_ARGS="--dataset fineweb \
  --model base \
  --batch_size $BATCH_SIZE \
  --acc_steps $ACC_STEPS \
  --iterations $ITERATIONS \
  --eval_interval $EVAL_INTERVAL \
  --sequence_length $SEQ_LEN \
  --n_layer $N_LAYER \
  --n_head $N_HEAD \
  --n_embd $N_EMBD \
  --device $DEVICE \
  --warmup_steps $WARMUP \
  --grad_clip $GRAD_CLIP \
  --results_base_folder $EXPS_DIR \
  --tensorboard"

SIGNMUON_SHARED="--beta1 $MUON_BETA1 --beta2 $MUON_BETA2 --momentum $MUON_MOM --nesterov True \
  --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY --scheduler $SM_SCHEDULER"

LIONMUON_SHARED="--beta1 $LM_BETA1 --beta2 $LM_BETA2 --scheduler $LM_SCHEDULER"

MUON_SHARED="--beta1 $MUON_BETA1 --beta2 $MUON_BETA2 --momentum $MUON_MOM --nesterov True \
  --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY --scheduler $MUON_SCHEDULER"

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
  if [ -f "${EXPS_DIR}/${name}/summary.json" ]; then
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
  OMP_NUM_THREADS=1 $PYTHON ./src/main.py $COMMON_ARGS \
    --experiment_name "$name" "$@" &
  PIDS+=($!)
}

# ── Muon (sign_muon K=1, pure NS every step, wsd scheduler) ─────────────────
MUON_LRS="1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 7e-3 1e-2 2e-2 3e-2"
for LR in $MUON_LRS; do
  run "muon_lr${LR}" \
    --opt sign_muon --lr $MUON_ADAMW_LR --muon_lr_factor $LR \
    --muon_every_k 1 --cheap_mode sign \
    $MUON_SHARED
done

# ── Signum (SignMuon K=inf: pure sign on 2D params, AdamW on 1D) ─────────────
SIGNUM_LRS="1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3"
for SLR in $SIGNUM_LRS; do
  run "signum_slr${SLR}" \
    --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_LR \
    --sign_lr $SLR --muon_every_k 10000000 --cheap_mode sign --sign_scaling none \
    $SIGNMUON_SHARED
done

# ── SignMuon ────────────────────────────────────────────────────────────────
# K=2  (best: lr=2e-3, slr=5e-6 — slr at lower boundary → extend down)
SM_K2_LRS="1e-3 2e-3 3e-3 5e-3 7e-3"
SM_K2_SLRS="1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4"
for LR in $SM_K2_LRS; do
  for SLR in $SM_K2_SLRS; do
    run "signmuon_k2_lr${LR}_slr${SLR}" \
      --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 2 --cheap_mode sign --sign_scaling none \
      $SIGNMUON_SHARED
  done
done

# K=5  (best: lr=3e-3, slr=2e-5 — both at lower boundary → extend down)
SM_K5_LRS="1e-3 2e-3 3e-3 5e-3 7e-3"
SM_K5_SLRS="5e-6 1e-5 2e-5 5e-5 1e-4"
for LR in $SM_K5_LRS; do
  for SLR in $SM_K5_SLRS; do
    run "signmuon_k5_lr${LR}_slr${SLR}" \
      --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 5 --cheap_mode sign --sign_scaling none \
      $SIGNMUON_SHARED
  done
done

# K=20  (best: lr=1e-2, slr=2e-5 — slr at lower boundary → extend down)
SM_K20_LRS="3e-3 5e-3 7e-3 1e-2 2e-2"
SM_K20_SLRS="5e-6 1e-5 2e-5 5e-5 1e-4"
for LR in $SM_K20_LRS; do
  for SLR in $SM_K20_SLRS; do
    run "signmuon_k20_lr${LR}_slr${SLR}" \
      --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 20 --cheap_mode sign --sign_scaling none \
      $SIGNMUON_SHARED
  done
done

# K=100  (best: lr=1e-2, slr=2e-5 — slr at lower boundary → extend down)
SM_K100_LRS="3e-3 5e-3 7e-3 1e-2 2e-2"
SM_K100_SLRS="5e-6 1e-5 2e-5 5e-5 1e-4"
for LR in $SM_K100_LRS; do
  for SLR in $SM_K100_SLRS; do
    run "signmuon_k100_lr${LR}_slr${SLR}" \
      --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 100 --cheap_mode sign --sign_scaling none \
      $SIGNMUON_SHARED
  done
done

# ── LionMuon ─────────────────────────────────────────────────────────────────
# K=1  (best: lr=1e-3 — interior optimum, resolved)
LM_K1_LRS="5e-4 7e-4 1e-3 2e-3 3e-3"
for LR in $LM_K1_LRS; do
  run "lionmuon_k1_lr${LR}" \
    --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
    --muon_every_k 1 --muon_ns_steps $MUON_NS_STEPS \
    --weight_decay $WEIGHT_DECAY \
    $LIONMUON_SHARED
done

# K=2  (best: lr=2e-3, slr=2e-5 — interior optimum, resolved)
LM_K2_LRS="1e-3 2e-3 3e-3 5e-3 7e-3"
LM_K2_SLRS="5e-6 1e-5 2e-5 5e-5 1e-4"
for LR in $LM_K2_LRS; do
  for SLR in $LM_K2_SLRS; do
    run "lionmuon_k2_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 2 \
      --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY \
      $LIONMUON_SHARED
  done
done

# K=5  (best: lr=5e-3, slr=5e-5 — interior optimum, resolved)
LM_K5_LRS="1e-3 2e-3 3e-3 5e-3 7e-3"
LM_K5_SLRS="5e-6 1e-5 2e-5 5e-5 1e-4"
for LR in $LM_K5_LRS; do
  for SLR in $LM_K5_SLRS; do
    run "lionmuon_k5_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 5 \
      --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY \
      $LIONMUON_SHARED
  done
done

# K=20  (best: lr=7e-3, slr=5e-5 — lr at upper boundary → extend up)
LM_K20_LRS="3e-3 5e-3 7e-3 1e-2 2e-2"
LM_K20_SLRS="2e-5 5e-5 1e-4"
for LR in $LM_K20_LRS; do
  for SLR in $LM_K20_SLRS; do
    run "lionmuon_k20_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 20 \
      --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY \
      $LIONMUON_SHARED
  done
done

# K=100  (best: lr=7e-3, slr=5e-5 — lr at upper boundary → extend up)
LM_K100_LRS="3e-3 5e-3 7e-3 1e-2 2e-2"
LM_K100_SLRS="2e-5 5e-5 1e-4"
for LR in $LM_K100_LRS; do
  for SLR in $LM_K100_SLRS; do
    run "lionmuon_k100_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 100 \
      --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY \
      $LIONMUON_SHARED
  done
done

# Wait for all remaining jobs
echo "Waiting for remaining jobs..."
wait
echo "Done!"
