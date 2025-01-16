#!/bin/bash

# Train DeepLabV2_ResNet101
# python3 sem_seg.py \
#     --train \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --source_domain Rural \
#     --horizontal_flip_augmentation \
#     --shift_scale_rotate_augmentation \
#     --brightness_contrast_augmentation \
#     --coarse_dropout_augmentation \
#     --batch_size 6 \
#     --criterion CrossEntropyLoss \
#     --optimizer SGD \
#     --scheduler PolynomialLR \
#     --epochs 20


# Resume DeepLabV2_ResNet101
# python3 sem_seg.py \
#     --train \
#     --resume \
#     --resume_epoch 14 \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --source_domain Rural \
#     --horizontal_flip_augmentation \
#     --shift_scale_rotate_augmentation \
#     --brightness_contrast_augmentation \
#     --coarse_dropout_augmentation \
#     --batch_size 6 \
#     --criterion CrossEntropyLoss \
#     --optimizer SGD \
#     --scheduler PolynomialLR \
#     --epochs 20


# Train PIDNet_S_Adversarial
# python3 sem_seg.py \
#     --train \
#     --model_name PIDNet_S \
#     --version T \
#     --batch_size 6 \
#     --epochs 30

# Train PIDNet_S_Adversarial
python3 sem_seg_da_adv.py \
    --train \
    --mode single_level \
    --model_name PIDNet_S \
    --version T \
    --horizontal_flip_augmentation \
    --shift_scale_rotate_augmentation \
    --grid_distortion_augmentation \
    --batch_size 6 \
    --epochs 30

# python3 sem_seg_da_adv.py \
#     --train \
#     --mode multi_level \
#     --model_name PIDNet_S \
#     --version 1 \
#     --horizontal_flip_augmentation \
#     --shift_scale_rotate_augmentation \
#     --batch_size 6 \
#     --epochs 30