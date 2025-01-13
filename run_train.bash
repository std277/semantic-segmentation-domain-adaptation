#!/bin/bash

# Train DeepLabV2_ResNet101
# python3 main.py \
#     --train \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --source_domain Rural \
#     --batch_size 6 \
#     --criterion CrossEntropyLoss \
#     --optimizer SGD \
#     --scheduler PolynomialLR \
#     --epochs 20


# Resume DeepLabV2_ResNet101
# python3 main.py \
#     --train \
#     --resume \
#     --resume_epoch 14 \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --source_domain Rural \
#     --batch_size 8 \
#     --optimizer SGD \
#     --scheduler PolynomialLR \
#     --lr 0.01 \
#     --power 0.6 \
#     --epochs 20




# Train PIDNet_S
python3 main.py \
    --train \
    --model_name PIDNet_S \
    --version 0 \
    --source_domain Rural \
    --data_augmentation \
    --batch_size 6 \
    --criterion CrossEntropyLoss \
    --optimizer SGD \
    --scheduler PolynomialLR \
    --epochs 20
