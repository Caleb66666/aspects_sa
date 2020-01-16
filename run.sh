#!/bin/bash
model_name=base_xlnet
file="$model_name".out
if [[ ! -f "$file" ]]; then
	touch "$file"
fi

nohup python -u train_infer.py \
            --pattern train \
            --model "$model_name" \
            --seed 279 \
            >"$file" 2>&1 &

tail -f "$file"
