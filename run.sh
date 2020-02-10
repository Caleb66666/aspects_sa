#!/usr/bin/env bash

model=bench_mark
out=nohup.out
if [[ ! -f "$out" ]]; then
    touch "$out"
fi

nohup python -u train_infer.py \
            --pattern train \
            --model "$model" \
            --seed 279 \
            --restore \
            &
tail -f "$out"
