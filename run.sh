#!/usr/bin/env bash

model=transfer_sfu
out=nohup.out
if [[ ! -f "$out" ]]; then
    touch "$out"
fi

nohup python -u train_infer.py \
            --pattern train \
            --model "$model" \
            --debug \
            &
tail -f "$out"
