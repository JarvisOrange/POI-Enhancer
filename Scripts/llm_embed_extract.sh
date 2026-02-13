#!/bin/bash

for variable1 in 'NY' 'SG' 'TKY'
    do
    for variable2 in   'address' 'time' 'cat_nearby'
    do  
        python Tools/get_embedding_from_LLM.py  --LLM llama2 --dataset $variable1 --gpu 3 --prompt_type $variable2
    done
done

