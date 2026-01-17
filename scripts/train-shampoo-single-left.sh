#!/bin/bash
cd /home/arman/llm-baselines
python ./src/main.py \
    --model base \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 16 \
    --sequence_length 512 \
    --acc_steps 4 \
    --dropout 0.05 \
    --dataset fineweb \
    --opt shampoo_single \
    --shampoo_side left \
    --lr 1e-3 \
    --wandb \
    --wandb_run_prefix shampoo_single_left
