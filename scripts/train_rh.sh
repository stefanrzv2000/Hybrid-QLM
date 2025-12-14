#!/bin/bash

dataset=MC

for seed in 1
do
    echo " "
    echo "Training MC qcnn seed $seed"
    bash scripts/train_qcnn.sh 33 $dataset 10 10 $seed --backend ibm_brussels --shots 1024 --debug
done

# dataset=TS

# for seed in 12345
# do
#     echo " "
#     echo "Training TS qcnn seed $seed"
#     bash scripts/train_qcnn.sh 33 $dataset 32 20 $seed --backend ibm_brussels --shots 256 --debug
# done