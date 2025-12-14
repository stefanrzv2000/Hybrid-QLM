#!/bin/bash

# dataset=MC

# for seed in 12 1234 12345
# do
#     echo " "
#     echo "Training MC qcnn seed $seed"
#     bash scripts/train_qcnn.sh 33 $dataset 10 10 $seed --backend ibm_brussels --shots 512 --debug
# done

dataset=TS

for seed in 12345
do
    echo " "
    echo "Training TS qrnn seed $seed"
    bash scripts/train_qrnn.sh $dataset 32 20 $seed --backend ibm_strasbourg --shots 256 --debug
done