#!/usr/bin/env bash

model=atae_lstm
out=nohup.out
if [[ ! -f "$out" ]]; then
    touch "$out"
fi

nohup python -u train_infer.py \
            --pattern train \
            --model "$model" \
            --restore \
            --seed 279 \
            &
tail -f "$out"
