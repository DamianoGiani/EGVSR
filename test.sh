#!/usr/bin/env bash

# This script is used to test pretrained models. More specific setttings can
# be found and modified in a test.yml file under the experiment dir

# basic settings
degradation=$1
model=$2
name=background
name1=foreground
gpu_id=0
exp_id=001


# run
python ./codes/main.py \
  --exp_dir ./experiments_${degradation}/${model}/${exp_id} \
  --mode test \
  --model ${model} \
  --opt test.yml \
  --gpu_id ${gpu_id}\
  --name ${name}

python ./codes/main.py \
  --exp_dir ./experiments_${degradation}/${model}/${exp_id} \
  --mode test \
  --model ${model} \
  --opt test1.yml \
  --gpu_id ${gpu_id}\
  --name ${name1}

python ./codes/main.py \
  --exp_dir ./experiments_${degradation}/${model}/${exp_id} \
  --mode test \
  --model ${model} \
  --opt test2.yml \
  --gpu_id ${gpu_id}\
