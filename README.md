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
| 0       | True     | Rural      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 100        | Rural         | 27.50    |
| 1       | True     | Rural      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         | Rural         |     |
| 2       | True     | Rural      | 6          | OhemCrossEntropyLoss | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         | Rural         |     |


Notes:


Next steps:
- PIDNet_S validation on test on which prediction (0 or 1) understand the model
- Loss function? Understand the (loss_s, loss_sb) contribution
- Check everything
- Fix OhemCrossEntropyLoss














## Old Trainings

Model name: `DeepLabV2_ResNet101`

Resizing: (512, 512)

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

