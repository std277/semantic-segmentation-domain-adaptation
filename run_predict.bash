#!/bin/bash

# Predict DeepLabV2_ResNet101
# python3 sem_seg.py \
#     --predict \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --model_file best_1.pt \
#     --target_domain Rural



# Predict PIDNet_S
python3 sem_seg.py \
    --predict \
    --model_name PIDNet_S \
    --version 6 \
    --model_file best_0.pt \
    --target_domain Urban


# Predict PIDNet_M
# python3 sem_seg.py \
#     --predict \
#     --model_name PIDNet_M \
#     --version 1 \
#     --model_file best_0.pt \
#     --target_domain Rural


# Predict PIDNet_S_Adversarial
# python3 sem_seg_da_adv.py \
#     --predict \
#     --model_name PIDNet_S \
#     --version 3 \
#     --model_file best_0.pt \
#     --target_domain Rural


# Predict PIDNet_S_DACS
# python3 sem_seg_dacs.py \
#     --predict \
#     --model_name PIDNet_S \
#     --version 3 \
#     --model_file best_0.pt \
#     --target_domain Rural


# Predict PIDNet_S_DACS_GCW_LDQ
# python3 sem_seg_dacs_gcw_ldq.py \
#     --predict \
#     --model_name PIDNet_S \
#     --version 3 \
#     --model_file best_0.pt \
#     --target_domain Rural








# Predict PIDNet_S_DACS X
# python3 sem_seg_dacs.py \
#     --predict \
#     --model_name PIDNet_S \
#     --version X \
#     --model_file best_2.pt \
#     --target_domain Rural


# Predict PIDNet_S_DACS_GCW_LDQ X
# python3 sem_seg_dacs_gcw_ldq.py \
#     --predict \
#     --model_name PIDNet_S \
#     --version X \
#     --model_file best_3.pt \
#     --target_domain Rural