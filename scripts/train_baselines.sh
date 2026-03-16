#!/bin/bash
set -e

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/common_config.sh"

PYTHON=/data/users/arman/miniconda3/envs/optim/bin/python

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [GPU_ID] [--dataset fw|fineweb|spj|slimpajama] [--model base|gpt|llama] [--max-jobs N]

Examples:
  $(basename "$0") 0 --dataset fw --model base
  $(basename "$0") 1 --dataset spj --model llama
  $(basename "$0") --dataset fineweb --model llama
EOF
}

GPU_ID="0"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  GPU_ID="$1"
  shift
fi

DATASET_RAW="fw"
MODEL_RAW="base"
MAX_JOBS_OVERRIDE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET_RAW="$2"
      shift 2
      ;;
    --model)
      MODEL_RAW="$2"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

case "$DATASET_RAW" in
  fw|fineweb)
    DATASET="fineweb"
    DATASET_PREFIX="fw"
    ;;
  spj|slimpajama)
    DATASET="slimpajama"
    DATASET_PREFIX="spj"
    ;;
  *)
    echo "Unsupported dataset: $DATASET_RAW"
    usage
    exit 1
    ;;
esac

case "$MODEL_RAW" in
  base|gpt)
    MODEL="base"
    MODEL_PREFIX="base"
    ;;
  llama)
    MODEL="llama"
    MODEL_PREFIX="llama"
    ;;
  *)
    echo "Unsupported model: $MODEL_RAW"
    usage
    exit 1
    ;;
esac

DEVICE="cuda:${GPU_ID}"

ITERATIONS=${ITERATIONS:-64000}
WARMUP=${WARMUP:-3000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
MAX_JOBS_LOCAL=${MAX_JOBS_OVERRIDE:-${MAX_JOBS:-1}}

EXPS_DIR=./exps
EXP_PREFIX="${DATASET_PREFIX}_${MODEL_PREFIX}_"

# Shared baseline hyperparameters for both base and llama.
ADAMW_LR=5e-4
ADAMW_BETA1=0.8
ADAMW_BETA2=0.999

LION_LR=1e-4
SIGNUM_LR=5e-5

MUON_LR=1e-3

SM_K2_LR=2e-3;   SM_K2_SLR=2e-5
SM_K5_LR=3e-3;   SM_K5_SLR=2e-5
SM_K20_LR=1e-2;  SM_K20_SLR=5e-5
SM_K100_LR=1e-2; SM_K100_SLR=5e-5

LMK1_LR=7e-4
LM_K2_LR=1e-3;   LM_K2_SLR=5e-5
LM_K5_LR=2e-3;   LM_K5_SLR=5e-5
LM_K20_LR=5e-3;  LM_K20_SLR=5e-5
LM_K100_LR=7e-3; LM_K100_SLR=5e-5

DATASET_DIR_ARG=""
if [ "$DATASET" != "fineweb" ]; then
  DATASET_DIR_ARG="--datasets_dir $DATASETS_DIR"
fi

COMMON_ARGS="--dataset $DATASET \
  $DATASET_DIR_ARG \
  --model $MODEL \
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
  local exp_name="${EXP_PREFIX}${name}"
  if [ -f "${EXPS_DIR}/${exp_name}/summary.json" ]; then
    echo "[SKIP] $exp_name"
    return
  fi
  reap
  while [ ${#PIDS[@]} -ge $MAX_JOBS_LOCAL ]; do
    sleep 1
    reap
  done
  echo "[RUN]  $exp_name"
  OMP_NUM_THREADS=1 $PYTHON ./src/main.py $COMMON_ARGS \
    --experiment_name "$exp_name" "$@" &
  PIDS+=($!)
}

run "muon" \
  --opt sign_muon --lr $MUON_ADAMW_LR --muon_lr_factor $MUON_LR \
  --muon_every_k 1 --cheap_mode sign \
  --momentum $MUON_MOM --nesterov True

run "adamw" \
  --opt adamw --lr $ADAMW_LR --beta1 $ADAMW_BETA1 --beta2 $ADAMW_BETA2

run "signmuon_k2" \
  --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K2_LR \
  --sign_lr $SM_K2_SLR --muon_every_k 2 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True

run "signmuon_k5" \
  --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K5_LR \
  --sign_lr $SM_K5_SLR --muon_every_k 5 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True

run "signmuon_k20" \
  --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K20_LR \
  --sign_lr $SM_K20_SLR --muon_every_k 20 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True

run "signmuon_k100" \
  --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_K100_LR \
  --sign_lr $SM_K100_SLR --muon_every_k 100 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True

run "signum" \
  --opt sign_muon --lr $SM_ADAMW_LR \
  --sign_lr $SIGNUM_LR --muon_every_k 10000000 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True

run "lionmuon_k1" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LMK1_LR \
  --muon_every_k 1 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

run "lionmuon_k2" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K2_LR \
  --sign_lr $LM_K2_SLR --muon_every_k 2 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

run "lionmuon_k5" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K5_LR \
  --sign_lr $LM_K5_SLR --muon_every_k 5 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

run "lionmuon_k20" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K20_LR \
  --sign_lr $LM_K20_SLR --muon_every_k 20 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

run "lionmuon_k100" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K100_LR \
  --sign_lr $LM_K100_SLR --muon_every_k 100 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

run "lion" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K100_LR \
  --sign_lr $LM_K100_SLR --muon_every_k 10000000 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

echo "Waiting for remaining jobs..."
wait
echo "Done!"