#!/usr/bin/bash

# This script is used to train the model using k-fold cross validation.

# The script takes 1 argument:
#   The path to the config file.

data_file=$1
model_file=$2
# check if the config file exists

if [ ! -f $data_file ]; then
    echo "Data file does not exist"
    exit 1
fi

# check if the model file exists

if [ ! -f $model_file ]; then
    echo "Model file does not exist"
    exit 1
fi

# run a loop to run training for the 5 folds
k=5
search_string="fold_1"
for i in $(seq 1 $k); do
    replacement_value="fold_$i"
    sed -i "s/$search_string/$replacement_value/g" $data_file
    sed -i "s/$search_string/$replacement_value/g" $model_file
    python $model_file
    search_string=$replacement_value
done
