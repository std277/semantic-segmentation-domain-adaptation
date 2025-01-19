#!/bin/bash

# Predict DeepLabV2_ResNet101
# python3 sem_seg.py \
#     --predict \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --model_file best_0.pt \
#     --target_domain Rural



# Predict PIDNet_S
python3 sem_seg.py \
    --predict \
    --model_name PIDNet_S \
    --version 2 \
    --model_file best_1.pt \
    --target_domain Rural



# Predict PIDNet_S_Adversarial
# python3 sem_seg_da_adv.py \
#     --predict \
#     --model_name PIDNet_S \
#     --version 0 \
#     --model_file best_0.pt \
#     --target_domain Rural