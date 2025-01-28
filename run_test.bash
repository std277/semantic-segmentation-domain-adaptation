#!/bin/bash

# Test DeepLabV2_ResNet101
# python3 sem_seg.py \
#     --test \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --model_file best_0.pt \
#     --target_domain Rural \
#     --batch_size 6


# Test PIDNet_S
python3 sem_seg.py \
    --test \
    --model_name PIDNet_S \
    --version 6 \
    --model_file best_0.pt \
    --target_domain Rural \
    --batch_size 6


# Test PIDNet_S_Adversarial
# python3 sem_seg_da_adv.py \
#     --test \
#     --model_name PIDNet_S \
#     --version 0 \
#     --model_file best_0.pt \
#     --target_domain Rural \
#     --batch_size 6