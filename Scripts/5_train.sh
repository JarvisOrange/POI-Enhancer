#!/bin/bash
python main.py \
    --simple_dataset False \
    --dataset NY \
    --poi_model hier \
    --LLM llama2 \
    --gpu 3 \
    --epoch_num 50 \
    --dim 256 \
    --batch_size 64 \
    --cross_layer_num 5 \
    --save_interval 10