# semantic-segmentation-domain-adaptation
Real-time Domain Adaptation in Semantic Segmentation

## Trainings

Model name: `DeepLabV2_ResNet101`

Resizing: (512, 512)

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                | SCHEDULER                           | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------|-------------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 22.88            | 19.93            |
| 1       | -                  | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 26.27            | 13.91            |






Model name: `PIDNet_S`

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                | SCHEDULER                           | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------|-------------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 20         | 39.31            | 31.12            |
| 1       | -                  | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 20         | 35.29            | 23.52            |
| 2       | -                  | Rural      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 41.97            | 33.16            |
| 3       | -                  | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 39.24            | 25.44            |
| 4       | (HF, SSR)          | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 40.38            | 27.50            |
| 5       | (HF, SSR, GD)      | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 38.04            | 27.00            |
| 6       | (HF, SSR, RC)      | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 38.94            | **28.30**        |
| 7       | (HF, SSR, CD)      | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | **41.11**        | 27.21            |
| 8       | (HF, RC, CD)       | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 38.14            | 25.19            |
| 9       | (HF, SSR, RC, CJ)  | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 37.65            | 24.30            |
| 10      | (HF, SSR, RC, GB)  | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 38.53            | 26.61            |

Data augmentation:
- HF: Horizontal Flip
- SSR: Shift Scale Rotate
- GD: Grid Distortion
- RC: Random Crop
- BC: Brightness Contrast
- CD: Coarse Dropout
- CJ: Color Jitter
- GB: Gaussian Blur

Model parameters: 7.718M

FLOPs: 0.142T

Mean inference time: 5.241 ms

Standard deviation of inference time: 2.625 ms

(Latencies computed on google colab with Tesla T4 GPU)






Model name: `PIDNet_S_DACS`

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                | SCHEDULER                           | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------|-------------------------------------|------------|------------------|------------------|
| 0       | -                  | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 34.12            | 16.52            |
| 1       | (RC)               | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 32.56            | 18.62            |
| 2       | (RC, CJ, GB)       | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         |             |             |
| 3       | (RC, HF, SSR)      | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         |             |             |
| 4       | (CJ, GB)           | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 37.48            | 19.54            |
| 5       | (HF, SSR)          | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 36.11            | 19.71            |


Data augmentation:
- HF: Horizontal Flip
- SSR: Shift Scale Rotate
- GD: Grid Distortion
- RC: Random Crop
- BC: Brightness Contrast
- CD: Coarse Dropout
- CJ: Color Jitter
- GB: Gaussian Blur

Model parameters: 7.718M

FLOPs: 0.142T

Mean inference time: 5.241 ms

Standard deviation of inference time: 2.625 ms

(Latencies computed on google colab with Tesla T4 GPU)

