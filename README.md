# semantic-segmentation-domain-adaptation
Real-time Domain Adaptation in Semantic Segmentation


## Trainings

Model name: `DeepLabV2_ResNet101`

| VERSION | DATA AUG | SRC DOMAIN | BATCH SIZE | OPTIMIZER                                         | SCHEDULER                        | NUM_EPOCHS | TARGET DOMAIN | mIoU (%) |
|---------|----------|------------|------------|---------------------------------------------------|----------------------------------|------------|---------------|----------|
| 0       | False    | Rural      | 4          | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.6) | 20         | Rural         | 20.77    |

FLOPs: 0.37T

Mean inference time: 1496.733 ms (Intel Core i7 11th)

Standard deviation of inference time: 50.338 ms



Note:
- Version 0 to resume from epoch 14 (stopped by colab)





Model name: `PIDNet_S`

Resizing: (512, 512)

| VERSION | DATA AUG | SRC DOMAIN | BATCH SIZE | OPTIMIZER                                         | SCHEDULER                        | NUM_EPOCHS | TARGET DOMAIN | mIoU (%) |
|---------|----------|------------|------------|---------------------------------------------------|----------------------------------|------------|---------------|----------|
| 0       | False    | Rural      | 8          | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.6) | 20         | Rural         | 34.02    |
| 1       | True     | Rural      | 8          | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.6) | 20         | Rural         | 34.33    |

FLOPs: 50.53G

Mean inference time: 70.013 ms (Intel Core i7 11th)

Standard deviation of inference time: 4.612 ms









Model name: `PIDNet_S`

No resizing: (1024, 1024)

| VERSION | DATA AUG | SRC DOMAIN | BATCH SIZE | OPTIMIZER                                         | SCHEDULER                        | NUM_EPOCHS | TARGET DOMAIN | mIoU (%) |
|---------|----------|------------|------------|---------------------------------------------------|----------------------------------|------------|---------------|----------|
| X       | True     | Rural      | 8          | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.001)  | PolynomialLR(lr=0.01, power=0.9) | 20         | Rural         | 31.86    |


FLOPs: 0.202T

Mean inference time: 353.668 ms

Standard deviation of inference time: 5.555 ms








Notes:
- Look for training in cityscapes
- Loss function: https://chatgpt.com/share/67829b87-b5d0-800f-8565-00127f6ed5bc
- Add loss function argument and adjust training logs and code to make logs