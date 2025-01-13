# semantic-segmentation-domain-adaptation
Real-time Domain Adaptation in Semantic Segmentation

## Trainings

Model name: `DeepLabV2_ResNet101`

Resizing: (512, 512)

| VERSION | DATA AUG | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                         | SCHEDULER                        | NUM_EPOCHS | TARGET DOMAIN | mIoU (%) |
|---------|----------|------------|------------|----------------------|---------------------------------------------------|----------------------------------|------------|---------------|----------|
| 0       | True     | Rural      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         | Rural         |     |



Model name: `PIDNet_S`

Resizing: (512, 512)

| VERSION | DATA AUG | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                         | SCHEDULER                        | NUM_EPOCHS | TARGET DOMAIN | mIoU (%) |
|---------|----------|------------|------------|----------------------|---------------------------------------------------|----------------------------------|------------|---------------|----------|
| 0       | True     | Rural      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         | Rural         |     |


Notes:
- Deeplabv2 version 0 training on colab
- PIDNet_S version 0 training locally

Next steps:
- Fix OhemCrossEntropyLoss
