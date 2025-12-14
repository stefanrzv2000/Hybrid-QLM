#!/bin/bash

type=$1
dataset=$2
backend=$3
shots=$4
model=$5

# all args from 6 onwards
extra_args=${@:6}

# if type == 22 then pad_seq_len = 4
if [ "$type" -eq 22 ]; then
    pad_seq_len=4
else
    pad_seq_len=6
fi

micromamba activate qiskit
python eval.py --model qcnn --cnn_type $type --dataset $dataset \
    --pad_seq_len $pad_seq_len --backend $backend --shots $shots \
    --load_model $model $extra_args 

# source ./scripts/eval_qcnn.sh 33 TS ibm_strasbourg 1024 ./models/TS/qcnn/aer_simulator_shotsNone/type_33_emb3_seq6_reps2/PGPE_lr0.1_BS32_EP20_pop8_sigma0.05_seed4/best_model.npz 2>logs5.txt

# source ./scripts/eval_qcnn.sh 33 MC-LM ibm_strasbourg 1024 ./models/MC-LM/qcnn/aer_simulator_shotsNone/type_33_emb3_seq6_reps2/PGPE_lr0.1_BS16_EP40_pop8_sigma0.05_seed1/best_model.npz 2>logs5.txt

# source ./scripts/eval_qcnn.sh 22 RP ibm_strasbourg 1024 ./models/RP/qcnn/type_22_emb3_seq4_reps2_PGPE_lr0.1_BS10_pop8_sigma0.05_aer_simulator_seed123/best_model.npz 2>logs5.txt
