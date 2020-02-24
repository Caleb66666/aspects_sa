#!/usr/bin/env bash

model=transfer_rcnn
out=nohup.out
if [[ ! -f "$out" ]]; then
    touch "$out"
fi

nohup python -u train_infer.py \
            --pattern train \
            --model "$model" \
            &
tail -f "$out"
