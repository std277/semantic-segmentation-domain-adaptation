#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p weights_pretrained

wget https://drive.google.com/file/d/1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v/view?usp=sharing

mv deeplab_resnet_pretrained_imagenet.pth weights_pretrained/deeplab_resnet_pretrained_imagenet.pth