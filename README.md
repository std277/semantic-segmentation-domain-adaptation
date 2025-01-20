# semantic-segmentation-domain-adaptation
Real-time Domain Adaptation in Semantic Segmentation

## Trainings

Model name: `DeepLabV2_ResNet101`

Resizing: (512, 512)

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                | SCHEDULER                           | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------|-------------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         |             |             |
| 1       | -                  | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         |             |             |






Model name: `PIDNet_S`

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                | SCHEDULER                           | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------|-------------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 20         | 39.31            | 31.12            |
| 1       | -                  | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 20         | 35.29            | 23.52            |
| 2       | -                  | Rural      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 41.97            | 33.16            |
| 3       | -                  | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 39.24            | 25.44            |
| 4       | (HF, SSR)          | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 40.38            | 27.50            |
| 5       | (HF, SSR, GD)      | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 38.04            | 27.00            |
| 6       | (HF, SSR, RC)      | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 38.94            | 28.30            |
| 7       | (HF, SSR, CD)      | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 41.11            | 27.21            |
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
| 1       | (CJ, GB)           | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 37.48            | 19.54            |
| 2       | (HF, SSR)          | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 36.11            | 19.71            |
| 3       | (HF, SSR, RC)      | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 35.89            | 17.93            |

| 1       | (CJ, GB)           | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         |             |             |
| 2       | (HF, SSR)          | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         |             |             |
| 3       | (HF, SSR, RC)      | Urban      | 6          | OhemCrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         |             |             |

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



























## Old Trainings

Model name: `DeepLabV2_ResNet101`

Resizing: (512, 512)

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                | SCHEDULER                           | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------|-------------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 20         | 17.85            | 22.42            |


Model parameters: 43.016M

FLOPs: 1.11T

Mean inference time: 4.032 ms

Standard deviation of inference time: 3.590 ms

(Latencies computed on google colab with Tesla T4 GPU)






Model name: `PIDNet_S`

Resizing: (512, 512)

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                | SCHEDULER                           | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------|-------------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 20         | 36.84            | 25.25            |
| 1       | -                  | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 20         | 37.89            | 24.13            |
| 2       | (HF, SSR)          | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 30         | 36.96            | 27.21            |
| 3       | (BC, CD)           | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 30         | 35.36            | 23.13            |
| 4       | (HF, SSR, BC, CD)  | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 30         | 33.14            | 19.21            |
| 5       | (HF, SSR, GD)      | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 30         | 38.52            | 27.45            |

Data augmentation:
- HF: Horizontal Flip
- SSR: Shift Scale Rotate
- BC: Brightness Contrast
- CD: Coarse Dropout
- GD: Grid Distortion

Model parameters: 7.718M

FLOPs: 37.902G

Mean inference time: 5.090 ms

Standard deviation of inference time: 3.142 msg

(Latencies computed on google colab with Tesla T4 GPU)








Model name: `PIDNet_S_DACS`

Resizing: (512, 512)

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                | SCHEDULER                           | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------|-------------------------------------|------------|------------------|------------------|
| 0       | -                  | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 30         | 33.31            | 18.67            |
| 1       | (CJ, GB)           | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 30         | 30.35            | 20.34            | 
| 2       | (CJ, GB)           | Urban      | 2          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.01, power=0.9)    | 30         | 29.24            | 19.36            | 
| 3       | (CJ, GB)           | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 30         | 33.33            | 20.59            | 
| 4       | (CJ, GB)           | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.00025, power=0.9) | 30         | 31.00            | 20.43            | 
| 5       | (HF, SSR, GD)      | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 30         | 32.71            | 19.66            | 


Data augmentation:
- HF: Horizontal Flip
- SSR: Shift Scale Rotate
- BC: Brightness Contrast
- CD: Coarse Dropout
- GD: Grid Distortion
- CJ: Color Jitter
- GB: Gaussian Blur

Model parameters: 7.718M

FLOPs: 37.902G

Mean inference time: 5.896 ms

Standard deviation of inference time: 2.660 ms

(Latencies computed on google colab with Tesla T4 GPU)