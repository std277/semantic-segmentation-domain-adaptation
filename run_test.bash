#!/bin/bash

# Test DeepLabV2_ResNet101

# python3 main.py \
#     --test \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --model_file last_0.pt \
#     --target_domain Rural \
#     --batch_size 8


# python3 main.py \
#     --test \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --model_file last_0.pt \
#     --target_domain Rural \
#     --batch_size 8








# Test PIDNet_S

# python3 main.py \
#     --test \
#     --model_name PIDNet_S \
#     --version T \
#     --model_file last_0.pt \
#     --target_domain Rural \
#     --batch_size 6


python3 main.py \
    --test \
    --model_name PIDNet_S \
    --version T \
    --model_file best_0.pt \
    --target_domain Rural \
    --batch_size 6