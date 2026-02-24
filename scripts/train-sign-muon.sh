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
  --wandb_project sign-muon-ablation-v3"

MUON_LR=0.02
ADAMW_LR=3e-4
SHARED="--beta1 0.9 --beta2 0.95 --momentum 0.95 --nesterov True \
  --muon_ns_steps 6 --weight_decay 0.1"

run() {
  local name=$1; shift
  echo "[RUN]  $name"
  python ./src/main.py $COMMON_ARGS $SHARED --experiment_name "$name" "$@"
}

# Baseline: pure Muon (K=1)
run "baseline_muon" \
  --opt muon --lr $ADAMW_LR --muon_lr_factor $MUON_LR

# Test all 4 cheap modes across K values
# sign/norm: no --sign_lr → auto-scales to lr*0.25 = 0.005
# cheap_ns/cached: cheap_lr defaults to muon LR
for MODE in sign norm cheap_ns cached; do
  for K in 2 5 10 20; do
    if [ "$MODE" = "sign" ] || [ "$MODE" = "norm" ]; then
      # Auto-scaled LR (cheap_lr = lr * 0.25)
      run "sign_muon_${MODE}_k${K}" \
        --opt sign_muon --lr $ADAMW_LR --muon_lr_factor $MUON_LR \
        --muon_every_k $K --cheap_mode $MODE
    else
      # cheap_ns and cached use same LR as Muon
      run "sign_muon_${MODE}_k${K}" \
        --opt sign_muon --lr $ADAMW_LR --muon_lr_factor $MUON_LR \
        --sign_lr $MUON_LR --muon_every_k $K --cheap_mode $MODE
    fi
  done
done

echo "Done!"
