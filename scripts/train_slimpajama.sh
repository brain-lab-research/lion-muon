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

ITERATIONS=64000
WARMUP=3000
EVAL_INTERVAL=500
MAX_JOBS=1

EXPS_DIR=./exps

# ── Llama-specific LRs (from exps_tuning_llama) ─────────────────────────────
ADAMW_LR=1e-3
ADAMW_BETA1=0.9
ADAMW_BETA2=0.95

LION_LR=1e-4
SIGNUM_LR=2e-4

# Muon (k=1)
MUON_LR=1e-3

# SignMuon per-k
SM_K2_LR=2e-3;   SM_K2_SLR=5e-5
SM_K5_LR=3e-3;   SM_K5_SLR=5e-5
SM_K20_LR=1e-2;  SM_K20_SLR=5e-5
SM_K100_LR=2e-2; SM_K100_SLR=5e-5

# LionMuon per-k
LMK1_LR=1e-3
LM_K2_LR=2e-3;   LM_K2_SLR=5e-5
LM_K5_LR=3e-3;   LM_K5_SLR=5e-5
LM_K20_LR=7e-3;  LM_K20_SLR=5e-5
LM_K100_LR=7e-3; LM_K100_SLR=5e-5

COMMON_ARGS="--dataset slimpajama \
  --datasets_dir $DATASETS_DIR \
  --model llama \
  --batch_size $BATCH_SIZE \
  --acc_steps $ACC_STEPS \
  --iterations $ITERATIONS \
  --eval_interval $EVAL_INTERVAL \
  --sequence_length $SEQ_LEN \
  --n_layer $N_LAYER \
  --n_head $N_HEAD \
  --n_embd $N_EMBD \
  --device $DEVICE \
  --scheduler $SCHEDULER \
  --warmup_steps $WARMUP \
  --weight_decay $WEIGHT_DECAY \
  --grad_clip $GRAD_CLIP \
  --muon_ns_steps $MUON_NS_STEPS \
  --results_base_folder $EXPS_DIR \
  --tensorboard"

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
  if [ -f "${EXPS_DIR}/${name}/summary.json" ]; then
    echo "[SKIP] $name"
    return
  fi
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

# ── Baselines ────────────────────────────────────────────────────────────────

run "spj_adamw" \
  --opt adamw --lr $ADAMW_LR --beta1 $ADAMW_BETA1 --beta2 $ADAMW_BETA2

run "spj_lion" \
  --opt lion --lr $LION_LR --beta1 $LION_BETA1 --beta2 $LION_BETA2

run "spj_signum" \
  --opt signum --lr $SIGNUM_LR --momentum $SIGNUM_MOM

# ── Muon (k=1, pure NS every step) ──────────────────────────────────────────

run "spj_muon" \
  --opt sign_muon --lr $MUON_ADAMW_LR --muon_lr_factor $MUON_LR \
  --muon_every_k 1 --cheap_mode sign \
  --momentum $MUON_MOM --nesterov True --beta1 $MUON_BETA1 --beta2 $MUON_BETA2

# ── SignMuon ─────────────────────────────────────────────────────────────────

run "spj_signmuon_k2" \
  --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K2_LR \
  --sign_lr $SM_K2_SLR --muon_every_k 2 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True --beta1 $SM_BETA1 --beta2 $SM_BETA2

# run "spj_signmuon_k2_no_nesterov" \
#   --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K2_LR \
#   --sign_lr $SM_K2_SLR --muon_every_k 2 --cheap_mode sign --sign_scaling none \
#   --momentum $SM_MOM --nesterov False --beta1 $SM_BETA1 --beta2 $SM_BETA2

run "spj_signmuon_k5" \
  --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K5_LR \
  --sign_lr $SM_K5_SLR --muon_every_k 5 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True --beta1 $SM_BETA1 --beta2 $SM_BETA2

# run "spj_signmuon_k5_no_nesterov" \
#   --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K5_LR \
#   --sign_lr $SM_K5_SLR --muon_every_k 5 --cheap_mode sign --sign_scaling none \
#   --momentum $SM_MOM --nesterov False --beta1 $SM_BETA1 --beta2 $SM_BETA2

run "spj_signmuon_k20" \
  --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K20_LR \
  --sign_lr $SM_K20_SLR --muon_every_k 20 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True --beta1 $SM_BETA1 --beta2 $SM_BETA2

# run "spj_signmuon_k20_no_nesterov" \
#   --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K20_LR \
#   --sign_lr $SM_K20_SLR --muon_every_k 20 --cheap_mode sign --sign_scaling none \
#   --momentum $SM_MOM --nesterov False --beta1 $SM_BETA1 --beta2 $SM_BETA2

run "spj_signmuon_k100" \
  --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K100_LR \
  --sign_lr $SM_K100_SLR --muon_every_k 100 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True --beta1 $SM_BETA1 --beta2 $SM_BETA2

# run "spj_signmuon_k100_no_nesterov" \
#   --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K100_LR \
#   --sign_lr $SM_K100_SLR --muon_every_k 100 --cheap_mode sign --sign_scaling none \
#   --momentum $SM_MOM --nesterov False --beta1 $SM_BETA1 --beta2 $SM_BETA2

# ── LionMuon ────────────────────────────────────────────────────────────────

# K=1: every step is Muon but with Lion dual EMA (β1=0.95,β2=0.99)
run "spj_lionmuon_k1" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LMK1_LR \
  --muon_every_k 1 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

run "spj_lionmuon_k2" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K2_LR \
  --sign_lr $LM_K2_SLR --muon_every_k 2 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

run "spj_lionmuon_k5" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K5_LR \
  --sign_lr $LM_K5_SLR --muon_every_k 5 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

run "spj_lionmuon_k20" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K20_LR \
  --sign_lr $LM_K20_SLR --muon_every_k 20 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

run "spj_lionmuon_k100" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K100_LR \
  --sign_lr $LM_K100_SLR --muon_every_k 100 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

echo "Waiting for remaining jobs..."
wait
echo "Done!"
