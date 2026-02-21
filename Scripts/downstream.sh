#!/bin/bash

embed_name="NY_llama2_tale_256_Epoch_5"
baseline_name='tale_256_ny'
dataset='NY'

echo $n


python Downstream/traj_next_pre.py    --gpu 3 --NAME $embed_name --dataset $dataset --POI_MODEL_NAME $baseline_name

echo $embed_name

echo "----------- task 1 done -----------"

python Downstream/poi_clf.py    --gpu 3 --NAME $embed_name --dataset $dataset --POI_MODEL_NAME $baseline_name

echo $embed_name

echo "----------- task 2 done -----------"


python Downstream/flow_next_pre.py    --gpu 3 --NAME $embed_name --dataset $dataset --POI_MODEL_NAME $baseline_name

echo $embed_name

echo "----------- task 3 done -----------"


python Downstream/poi_cluster.py    --gpu 3 --NAME $embed_name --dataset $dataset --POI_MODEL_NAME $baseline_name

echo $embed_name


echo "----------- task 4 done -----------"
