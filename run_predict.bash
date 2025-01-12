#!/bin/bash

# Predict DeepLabV2_ResNet101
# python3 main.py \
#     --predict \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --model_file best_0.pt \
#     --target_domain Rural



# Predict PIDNet_S
python3 main.py \
    --predict \
    --model_name PIDNet_S \
    --version 1 \
    --model_file best_0.pt \
    --target_domain Rural