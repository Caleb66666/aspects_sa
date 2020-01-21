#!/bin/bash
model_name=albert_attn_pool
file="$model_name".out
if [[ ! -f "$file" ]]; then
	touch "$file"
fi

nohup python -u train_infer.py \
            --pattern train \
            --model "$model_name" \
            --seed 279 \
			--restore \
            >"$file" 2>&1 &

tail -f "$file"
