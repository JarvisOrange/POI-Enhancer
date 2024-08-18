#!/bin/bash

pn='poi2vec_256_sg'
dataset='SG'
gpu=2
p=0
n='Llama2'


echo $n

# python Downstream/llm_traj_next_pre.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn --epoch  50 --prompt $p

echo "----------- task 2 done -----------"

echo $n

python Downstream/llm_flow_next_pre.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn --prompt $p



echo "----------- task 3 done -----------"


# python Downstream/llm_traj_user_clf.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn --prompt $p
#!/bin/bash

