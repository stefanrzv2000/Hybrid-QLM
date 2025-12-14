#!/bin/bash

dataset=$1
BS=$2
epochs=$3
seed=$4
# all args from 5 onwards
extra_args=${@:5}

micromamba activate qiskit
python train.py --model qrnn --dataset $dataset --seed $seed --max_batches 10 \
    --epochs $epochs --save_model --batch_size $BS $extra_args
