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



# python3 sem_seg.py \
#     --train \
#     --model_name PIDNet_S \
#     --version 0 \
#     --source_domain Rural \
#     --batch_size 6 \
#     --criterion OhemCrossEntropyLoss \
#     --optimizer SGD \
#     --scheduler PolynomialLR \
#     --epochs 20


# python3 sem_seg.py \
#     --train \
#     --model_name PIDNet_S \
#     --version 6 \
#     --source_domain Urban \
#     --shift_scale_rotate_augmentation \
#     --random_crop_augmentation \
#     --batch_size 6 \
#     --criterion OhemCrossEntropyLoss \
#     --optimizer SGD \
#     --scheduler PolynomialLR \
#     --lr 0.001 \
#     --epochs 20
    # --coarse_dropout_augmentation \

















# Train PIDNet_S_Adversarial

# Train PIDNet_S_Adversarial
# python3 sem_seg_da_adv.py \
#     --train \
#     --mode single_level \
#     --model_name PIDNet_S \
#     --version T \
#     --horizontal_flip_augmentation \
#     --shift_scale_rotate_augmentation \
#     --grid_distortion_augmentation \
#     --batch_size 6 \
#     --epochs 30

# python3 sem_seg_da_adv.py \
#     --train \
#     --mode single_level \
#     --model_name PIDNet_S \
#     --version T \
#     --horizontal_flip_augmentation \
#     --shift_scale_rotate_augmentation \
#     --batch_size 6 \
#     --epochs 20

#     --random_crop_augmentation \


# python3 sem_seg_dacs.py \
#     --train \
#     --model_name PIDNet_S \
#     --version T \
#     --horizontal_flip_augmentation \
#     --shift_scale_rotate_augmentation \
#     --random_crop_augmentation \
#     --batch_size 2 \
#     --patience 10 \
#     --epochs 30

    # --grid_distortion_augmentation \
    # --color_jitter_augmentation \
    # --gaussian_blur_augmentation \




python3 sem_seg_dacs_gcw_ldq.py \
    --train \
    --model_name PIDNet_S \
    --version T \
    --gcw \
    --ldq \
    --augment_mixed \
    --horizontal_flip_augmentation \
    --shift_scale_rotate_augmentation \
    --random_crop_augmentation \
    --batch_size 2 \
    --epochs 30