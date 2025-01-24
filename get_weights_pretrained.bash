#!/bin/bash

# Create weights_pretrained directory if it doesn't exist
mkdir -p weights_pretrained

# Download DeepLabV2_Resnet101 weights pretrained on ImageNet
gdown "https://drive.google.com/uc?id=1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v" -O "weights_pretrained/deeplab_resnet_pretrained_imagenet.pth"

# Download PIDNet weights pretrained on ImageNet

gdown "https://drive.google.com/uc?id=1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-" -O "weights_pretrained/pidnet_s_pretrained_imagenet.pth"
gdown "https://drive.google.com/uc?id=1gB9RxYVbdwi9eO5lbT073q-vRoncpYT1" -O "weights_pretrained/pidnet_m_pretrained_imagenet.pth"
gdown "https://drive.google.com/uc?id=1Eg6BwEsnu3AkKLO8lrKsoZ8AOEb2KZHY" -O "weights_pretrained/pidnet_l_pretrained_imagenet.pth"