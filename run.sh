#!/usr/bin/env bash

model=transformer_attn_pool
out=nohup.out
if [[ ! -f "$out" ]]; then
    touch "$out"
fi

nohup python -u train_infer.py \
            --pattern train \
            --model "$model" \
#            --debug \
            --seed 279 \
            &
tail -f "$out"
