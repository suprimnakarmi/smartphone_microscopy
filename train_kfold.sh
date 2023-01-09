#!/usr/bin/bash

# This script is used to train the model using k-fold cross validation.

# The script takes 1 argument:
#   The path to the config file.

config_file=$1

# check if the config file exists

if [ ! -f $config_file ]; then
    echo "Config file does not exist"
    exit 1
fi

# run a loop to run training for the 5 folds

for i in {0..4}; do
    python mmdetection/tools/train.py $config_file --gpu-id 1 --cfg-options fold=$i
done