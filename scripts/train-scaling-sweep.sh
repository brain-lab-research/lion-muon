#!/bin/bash
set -e

export OMP_NUM_THREADS=1

cd /home/arman.bolatov/Desktop/llm-baselines

DEVICE="cuda:1"
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
  --wandb_project sign-muon-scaling"

run() {
  local name=$1; shift
  echo "[RUN]  $name"
  python ./src/main.py $COMMON_ARGS --experiment_name "$name" "$@"
}

# Adam baseline
for LR in 1e-3 3e-4 1e-4 3e-5; do
  run "adam_lr${LR}" \
    --opt adamw --lr $LR --beta1 0.9 --beta2 0.95 --weight_decay 0.1
done

# sign_k5: 3 scaling modes x 4 LRs
# sign_lr is set explicitly for each
for SCALING in muon frob none; do
  for SLR in 0.001 0.005 0.01 0.02; do
    run "sign_k5_${SCALING}_slr${SLR}" \
      --opt sign_muon --lr 3e-4 --muon_lr_factor 0.02 \
      --sign_lr $SLR --muon_every_k 5 --cheap_mode sign \
      --sign_scaling $SCALING \
      --beta1 0.9 --beta2 0.95 --momentum 0.95 --nesterov True \
      --muon_ns_steps 6 --weight_decay 0.1
  done
done

echo "Done!"
