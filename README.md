# semantic-segmentation-domain-adaptation
Real-time Domain Adaptation in Semantic Segmentation

## Trainings

Model name: `DeepLabV2_ResNet101`

Resizing: (512, 512)

| VERSION | DATA AUG | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                         | SCHEDULER                        | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|----------|------------|------------|----------------------|---------------------------------------------------|----------------------------------|------------|------------------|------------------|
| 0       | -        | Rural      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         | 17.85            | 22.42            |


Model parameters: 43016284

FLOPs: 1.11T

Mean inference time: 4.032 ms

Standard deviation of inference time: 3.590 ms

(Latencies computed on google colab with Tesla T4 GPU)






Model name: `PIDNet_S`

Resizing: (512, 512)

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                         | SCHEDULER                        | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|---------------------------------------------------|----------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         | 36.84            | 25.25            |
| 1       | -                  | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         | 37.89            | 24.13            |
| 2       | (HF, SSR)          | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         |             |             |
| 3       | (BC, CD)           | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         |             |             |
| 4       | (HF, SSR, BC, CD)  | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 20         |             |             |

Data augmentation:
- HF: Horizontal Flip
- SSR: Shift Scale Rotate
- BC: Brightness Contrast
- CD: Coarse Dropout

Model parameters: 7717839

FLOPs: 37.902G

Mean inference time: 5.090 ms

Standard deviation of inference time: 3.142 ms

(Latencies computed on google colab with Tesla T4 GPU)





Current steps:
- PIDNet (2,3,4) training on colab

Next steps:
- Accuracy for each class to compare with class balance
- Fix OhemCrossEntropyLoss
