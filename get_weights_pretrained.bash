#!/bin/bash

# Create weights_pretrained directory if it doesn't exist
mkdir -p weights_pretrained

gdown "https://drive.google.com/uc?id=1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v" -O "weights_pretrained/deeplab_resnet_pretrained_imagenet.pth"