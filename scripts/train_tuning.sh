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
  $(basename "$0") [GPU_ID] [--dataset fw|fineweb|spj|slimpajama] [--model base|gpt|llama] [--max-jobs N] [--results-dir DIR]

Examples:
  $(basename "$0") 0 --dataset fw --model base
  $(basename "$0") 0 --dataset fw --model llama
  $(basename "$0") --dataset spj --model llama --max-jobs 2
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
RESULTS_DIR_OVERRIDE=""

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
    --results-dir)
      RESULTS_DIR_OVERRIDE="$2"
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
    MODEL_PREFIX="gpt"
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
MAX_JOBS_LOCAL=${MAX_JOBS_OVERRIDE:-2}
SM_LR=${SM_LR:-1e-3}

if [ "$MODEL" = "base" ]; then
  ITERATIONS=2000
  WARMUP=200
  EVAL_INTERVAL=100
else
  ITERATIONS=3000
  WARMUP=300
  EVAL_INTERVAL=500
fi

if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
  EXPS_DIR="$RESULTS_DIR_OVERRIDE"
elif [ "$DATASET" = "fineweb" ] && [ "$MODEL" = "base" ]; then
  EXPS_DIR=./exps_tuning_gpt
elif [ "$DATASET" = "fineweb" ] && [ "$MODEL" = "llama" ]; then
  EXPS_DIR=./exps_tuning_llama
else
  EXPS_DIR="./exps_tuning_${DATASET_PREFIX}_${MODEL_PREFIX}"
fi

COMMON_ARGS="--dataset $DATASET \
  --datasets_dir $DATASETS_DIR \
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
  --grad_clip $GRAD_CLIP \
  --results_base_folder $EXPS_DIR \
  --tensorboard"

MUON_SHARED="--momentum $MUON_MOM --nesterov True \
  --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY"

SIGNMUON_SHARED="--momentum $SM_MOM --nesterov True \
  --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY"

LIONMUON_SHARED="--beta1 $LM_BETA1 --beta2 $LM_BETA2 \
  --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY"

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
  while [ ${#PIDS[@]} -ge $MAX_JOBS_LOCAL ]; do
    sleep 1
    reap
  done
  echo "[RUN]  $name"
  OMP_NUM_THREADS=1 $PYTHON ./src/main.py $COMMON_ARGS \
    --experiment_name "$name" "$@" &
  PIDS+=($!)
}

MUON_LRS="1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 7e-3 1e-2 2e-2 3e-2"
for LR in $MUON_LRS; do
  run "muon_lr${LR}" \
    --opt sign_muon --lr $MUON_ADAMW_LR --muon_lr_factor $LR \
    --muon_every_k 1 --cheap_mode sign \
    $MUON_SHARED
done

SIGNUM_LRS="1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3"
for SLR in $SIGNUM_LRS; do
  run "signum_slr${SLR}" \
    --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $SM_LR \
    --sign_lr $SLR --muon_every_k 10000000 --cheap_mode sign --sign_scaling none \
    $SIGNMUON_SHARED
done

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

LM_K1_LRS="5e-4 7e-4 1e-3 2e-3 3e-3"
for LR in $LM_K1_LRS; do
  run "lionmuon_k1_lr${LR}" \
    --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
    --muon_every_k 1 \
    $LIONMUON_SHARED
done

if [ "$MODEL" = "llama" ]; then
  LM_K2_LRS="5e-4 7e-4 1e-3 2e-3 3e-3 5e-3 7e-3"
else
  LM_K2_LRS="1e-3 2e-3 3e-3 5e-3 7e-3"
fi
LM_K2_SLRS="5e-6 1e-5 2e-5 5e-5 1e-4"
for LR in $LM_K2_LRS; do
  for SLR in $LM_K2_SLRS; do
    run "lionmuon_k2_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 2 \
      $LIONMUON_SHARED
  done
done

LM_K5_LRS="1e-3 2e-3 3e-3 5e-3 7e-3"
LM_K5_SLRS="5e-6 1e-5 2e-5 5e-5 1e-4"
for LR in $LM_K5_LRS; do
  for SLR in $LM_K5_SLRS; do
    run "lionmuon_k5_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 5 \
      $LIONMUON_SHARED
  done
done

LM_K20_LRS="3e-3 5e-3 7e-3 1e-2 2e-2"
LM_K20_SLRS="2e-5 5e-5 1e-4"
for LR in $LM_K20_LRS; do
  for SLR in $LM_K20_SLRS; do
    run "lionmuon_k20_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 20 \
      $LIONMUON_SHARED
  done
done

LM_K100_LRS="3e-3 5e-3 7e-3 1e-2 2e-2"
LM_K100_SLRS="2e-5 5e-5 1e-4"
for LR in $LM_K100_LRS; do
  for SLR in $LM_K100_SLRS; do
    run "lionmuon_k100_lr${LR}_slr${SLR}" \
      --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LR \
      --sign_lr $SLR --muon_every_k 100 \
      $LIONMUON_SHARED
  done
done

echo "Waiting for remaining jobs..."
wait
echo "Done!"
