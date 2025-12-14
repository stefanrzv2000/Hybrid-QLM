#!/bin/bash

# dataset=MC

# for seed in 2 3 4 5 6 7 8 9 10
# do
#     echo " "
#     echo "Training MC qcnn seed $seed"
#     bash scripts/train_qcnn.sh 33 $dataset 10 40 $seed
# done

# seed=1
# for shots in 4096 2048 1024 512 256 None
# do
#     echo " "
#     echo "Training MC qcnn shots $shots"
#     bash scripts/train_qcnn.sh 33 $dataset 10 10 $seed --shots $shots
# done

# dataset=RP

# for seed in 1 12 1234 12345
# do
#     echo " "
#     echo "Training RP qcnn seed $seed"
#     bash scripts/train_qcnn.sh 22 $dataset 10 40 $seed
# done

# for seed in 1 12 123 1234 12345
# do
#     echo " "
#     echo "Training RP qrnn seed $seed"
#     bash scripts/train_qrnn.sh $dataset 10 40 $seed
# done

# dataset=MC-LM

# for seed in 1 12 123 1234 12345
# do
#     echo " "
#     echo "Training MC-LM qcnn seed $seed"
#     bash scripts/train_qcnn.sh 33 $dataset 16 40 $seed
# done

# for seed in 1 12 123 1234 12345
# do
#     echo " "
#     echo "Training MC-LM qcnn seed $seed"
#     bash scripts/train_qcnn.sh 22 $dataset 16 40 $seed
# done

dataset=TS

# for seed in 1 12 123 1234 12345
# do
#     echo " "
#     echo "Training TS qrnn seed $seed"
#     bash scripts/train_qrnn.sh $dataset 32 30 $seed --alg SPSA
# done

seed=111
for emb in 2 3 4 5
do
    echo " "
    echo "Training TS qrnn embedding $emb"
    bash scripts/train_qrnn.sh $dataset 32 30 $seed --emb_size $emb
done

# seed=12345
# for shots in 256 512 1024 2048 4096 None
# do
#     echo " "
#     echo "Training TS qrnn shots $shots"
#     bash scripts/train_qrnn.sh $dataset 32 30 12345 --shots $shots
# done

# for seed in 12345 2 3 4 5
# do
#     echo " "
#     echo "Training TS qcnn seed $seed"
#     bash scripts/train_qcnn.sh 33 $dataset 32 20 $seed
# done