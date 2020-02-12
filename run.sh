#!/usr/bin/env bash

model=fix_len_model
out=nohup.out
if [[ ! -f "$out" ]]; then
    touch "$out"
fi

nohup python -u train_infer.py \
            --pattern train \
            --model "$model" \
            --seed 279 \
            &
tail -f "$out"
