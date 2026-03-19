#!/bin/bash
set -e

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/common_config.sh"

PYTHON=python

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [GPU_ID] [--dataset fw|fineweb|spj|slimpajama] [--model base|gpt|llama] [--max-jobs N]

Examples:
  $(basename "$0") 0 --dataset fw --model base
  $(basename "$0") 0 --dataset fw --model llama
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

# Match baseline experiment settings (same family as signmuon/lionmuon runs).
ITERATIONS=${ITERATIONS:-64000}
WARMUP=${WARMUP:-3000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
MAX_JOBS_LOCAL=${MAX_JOBS_OVERRIDE:-${MAX_JOBS:-1}}

EXPS_DIR=./exps
EXP_PREFIX="${DATASET_PREFIX}_${MODEL_PREFIX}_"

# AdamW settings from train_baselines.sh
ADAMW_LR=5e-4
ADAMW_BETA1=0.8
ADAMW_BETA2=0.999

# Best tuned LR per RMSspectral variant from exps_tuning_rmsspectral_fw_gpt
RMS_PRE_LR=2e-2
RMS_PRE_EMA_LR=2e-2
RMS_POST_ORTH_LR=5e-4
RMS_SPLIT_LR=1e-4
RMS_BETA1=${RMS_BETA1:-0.9}
RMS_BETA2=${RMS_BETA2:-0.95}

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

# run "adamw" \
#   --opt adamw --lr $ADAMW_LR --beta1 $ADAMW_BETA1 --beta2 $ADAMW_BETA2

run "rmsspectral_pre" \
  --opt rmsspectral --lr $RMS_PRE_LR --beta1 $RMS_BETA1 --beta2 $RMS_BETA2 \
  --rmsspectral_variant pre

run "rmsspectral_pre_ema" \
  --opt rmsspectral --lr $RMS_PRE_EMA_LR --beta1 $RMS_BETA1 --beta2 $RMS_BETA2 \
  --rmsspectral_variant pre_ema

run "rmsspectral_post_orth" \
  --opt rmsspectral --lr $RMS_POST_ORTH_LR --beta1 $RMS_BETA1 --beta2 $RMS_BETA2 \
  --rmsspectral_variant post_orth

run "rmsspectral_split" \
  --opt rmsspectral --lr $RMS_SPLIT_LR --beta1 $RMS_BETA1 --beta2 $RMS_BETA2 \
  --rmsspectral_variant split

echo "Waiting for remaining jobs..."
wait
echo "Done!"
