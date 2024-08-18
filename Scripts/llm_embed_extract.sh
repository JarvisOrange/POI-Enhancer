#!/bin/bash


for variable1 in 'NY' 'TKY' 'SG'
    do
    for variable2 in    'chatglm3'
    do  
        python Tools/get_embedding_from_LLM.py  --LLM $variable2 --dataset $variable1 --gpu 1 --prompt_type 'address'
    done
done


