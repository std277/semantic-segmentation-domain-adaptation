# semantic-segmentation-domain-adaptation
Real-time Domain Adaptation in Semantic Segmentation


## Trainings

Model name: DeepLabV2_ResNet101

| VERSION | DATA AUG | SRC DOMAIN | BATCH SIZE | OPTIMIZER                                         | SCHEDULER                        | NUM_EPOCHS | TARGET DOMAIN | mIoU (%) | INFERENCE TIME (ms) |
|---------|----------|------------|------------|---------------------------------------------------|----------------------------------|------------|---------------|----------|---------------------|
| 0       | False    | Rural      | 4          | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.6) | 20         | Rural         | 20.77    | 1437.69             |

Note:
- Version 0 to resume from epoch 14 (stopped by colab)


Model name: PIDNet_S

| VERSION | DATA AUG | SRC DOMAIN | BATCH SIZE | OPTIMIZER                                         | SCHEDULER                        | NUM_EPOCHS | TARGET DOMAIN | mIoU (%) | INFERENCE TIME (ms) |
|---------|----------|------------|------------|---------------------------------------------------|----------------------------------|------------|---------------|----------|---------------------|
| 0       | False    | Rural      | 8          | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.6) | 20         | Rural         | 34.02    | 73.32               |
| 1       | True     | Rural      | 8          | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.6) | 20         | Rural         | xx.xx    | xx.xx               |

Note:
- Version 1 training with data augmentation locally