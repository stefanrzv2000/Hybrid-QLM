#!/bin/bash

dataset=$1
backend=$2
shots=$3
model=$4

# all args from 5 onwards
extra_args=${@:5}

micromamba activate qiskit
python eval.py --model qrnn --dataset $dataset \
    --backend $backend --shots $shots --batch_size 32 --max_batches 10 \
    --load_model $model $extra_args

# source ./scripts/eval_qrnn.sh TS ibm_strasbourg 1024 ./models/TS/qrnn/emb3_seq6_PGPE_lr0.1_BS32_EP30_pop8_sigma0.05_aer_simulator_seed12345/best_model.npz 2>logs5.txt

# source ./scripts/eval_qrnn.sh MC-LM ibm_strasbourg 1024 ./models/MC-LM/qrnn/emb3_seq4_PGPE_lr0.1_BS16_EP40_pop8_sigma0.05_aer_simulator_seed12/best_model.npz 2>logs5.txt

# source ./scripts/eval_qrnn.sh RP ibm_strasbourg 1024 ./models/RP/qrnn/emb3_seq4_PGPE_lr0.1_BS10_EP40_pop8_sigma0.05_aer_simulator_seed1/best_model.npz 2>logs5.txt