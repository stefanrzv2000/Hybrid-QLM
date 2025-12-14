#!/bin/bash

type=$1
dataset=$2
BS=$3
epochs=$4
seed=$5
# all args from 6 onwards
extra_args=${@:6}

# if type == 22 then pad_seq_len = 4
if [ "$type" -eq 22 ]; then
    pad_seq_len=4
else
    pad_seq_len=6
fi

micromamba activate qiskit
python train.py --model qcnn --cnn_type $type --dataset $dataset --seed $seed --max_batches 10 \
    --epochs $epochs --save_model --batch_size $BS --pad_seq_len $pad_seq_len $extra_args

# source scripts/train_qcnn.sh 33 MC 10 10 321 --load_model ./models/MC/qcnn/ibm_brussels_shots1024/type_33_emb3_seq6_reps2/PGPE_lr0.1_BS10_EP10_pop8_sigma0.05_seed1/best_model.npz