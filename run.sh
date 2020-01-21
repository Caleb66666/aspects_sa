#!/usr/bin/env bash

model=albert_attn_pool
nohup python -u train_infer.py \
            --pattern train \
            --model "$model" \
            --seed 279 \
            >/dev/null 2>&1 &

