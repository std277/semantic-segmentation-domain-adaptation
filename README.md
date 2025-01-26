# semantic-segmentation-domain-adaptation
Real-time Domain Adaptation in Semantic Segmentation

## Trainings

### DeepLabV2_ResNet101

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                | SCHEDULER                           | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------|-------------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 22.88            | 19.93            |
| 1       | -                  | Urban      | 6          | CrossEntropyLoss     | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)   | 20         | 26.27            | 13.91            |


Model parameters: 43.016M

FLOPs: 1.11T

Mean inference time: 4.040 ms

Standard deviation of inference time: 2.718 ms

(Latencies computed on google colab with Tesla T4 GPU)



### PIDNet_S

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

Model parameters: 7.718M

FLOPs: 0.142T

Mean inference time: 5.241 ms

Standard deviation of inference time: 2.625 ms

(Latencies computed on google colab with Tesla T4 GPU)





### PIDNet_S_Adversarial

| VERSION | MODE         | DATA AUG      | SRC DOMAIN | BATCH SIZE | CRITERION            | CRITERION (D) | OPTIMIZER                               | OPTIMIZER (D)            | SCHEDULER                         | SCHEDULER (D)                      | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------|---------------|------------|------------|----------------------|---------------|-----------------------------------------|--------------------------|-----------------------------------|------------------------------------|------------|------------------|------------------|
| 0       | single_level | -             | Urban      | 6          | OhemCrossEntropyLoss | BCELoss       | SGD(momentum: 0.9 weight_decay: 0.0005) | Adam(betas: (0.9, 0.99)) | PolynomialLR(lr=0.001, power=0.9) | PolynomialLR(lr=0.0005, power=0.9) | 20         | 34.15            | 21.56            |
| 1       | single_level | (HF, SSR, RC) | Urban      | 6          | OhemCrossEntropyLoss | BCELoss       | SGD(momentum: 0.9 weight_decay: 0.0005) | Adam(betas: (0.9, 0.99)) | PolynomialLR(lr=0.001, power=0.9) | PolynomialLR(lr=0.0005, power=0.9) | 20         | 34.19            | 20.14            |
| 2       | multi_level  | -             | Urban      | 6          | OhemCrossEntropyLoss | BCELoss       | SGD(momentum: 0.9 weight_decay: 0.0005) | Adam(betas: (0.9, 0.99)) | PolynomialLR(lr=0.001, power=0.9) | PolynomialLR(lr=0.0005, power=0.9) | 20         | 33.68            | 17.74            |
| 3       | multi_level  | (HF, SSR, RC) | Urban      | 6          | OhemCrossEntropyLoss | BCELoss       | SGD(momentum: 0.9 weight_decay: 0.0005) | Adam(betas: (0.9, 0.99)) | PolynomialLR(lr=0.001, power=0.9) | PolynomialLR(lr=0.0005, power=0.9) | 20         | 35.41            | 20.59            |





### PIDNet_S_DACS

| VERSION | DATA AUG (DA-TARGET)      | SRC DOMAIN | BATCH SIZE | CRITERION        | OPTIMIZER                                | SCHEDULER                          | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|---------------------------|------------|------------|------------------|------------------------------------------|------------------------------------|------------|------------------|------------------|
| 0       | -                         | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 35.50            | 17.79            |
| 1       | (RC-ALL)                  | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 35.26            | 20.83            |
| 2       | (RC-ALL, CJ-MXD, GB-MXD)  | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 36.20            | 21.48            |
| 3       | (RC-ALL, HF-MXD, SSR-MXD) | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 35.82            | 21.82            |
| 4       | (RC-ALL, CJ-ALL, GB-ALL)  | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 31.39            | 17.54            |
| 5       | (RC-ALL, HF-ALL, SSR-ALL) | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 33.19            | 19.72            |

| X       | (RC-ALL, HF-MXD, SSR-MXD) | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.0001, power=0.9) | 200        | 35.10            | 20.07            |



### PIDNet_S_DACS_GCW_LDQ

| VERSION | TECHNIQUES   | DATA AUG (DA-TARGET)      | SRC DOMAIN | BATCH SIZE | CRITERION        | OPTIMIZER                                | SCHEDULER                          | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------|---------------------------|------------|------------|------------------|------------------------------------------|------------------------------------|------------|------------------|------------------|
| 0       | (GCW)        | (RC-ALL)                  | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 34.23            | 18.98            |
| 1       | (LDQ)        | (RC-ALL)                  | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 33.44            | 17.44            |
| 2       | (GCW, LDQ)   | (RC-ALL)                  | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 32.81            | 17.42            |
| 3       | (GCW)        | (RC-ALL, HF-MXD, SSR-MXD) | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 34.13            | 19.11            |
| 4       | (LDQ)        | (RC-ALL, HF-MXD, SSR-MXD) | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 32.34            | 16.04            |
| 5       | (GCW, LDQ)   | (RC-ALL, HF-MXD, SSR-MXD) | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 20         | 34.11            | 17.48            |

| X       | (GCW, LDQ)   | (RC-ALL, HF-MXD, SSR-MXD) | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.0001, power=0.9) | 200        | 35.09            | 19.32            |





### PIDNet_M experiments

| TECHNIQUES   | VERSION | DATA AUG (DA-TARGET)      | SRC DOMAIN | BATCH SIZE | CRITERION        | OPTIMIZER                                | SCHEDULER                          | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|--------------|---------|---------------------------|------------|------------|------------------|------------------------------------------|------------------------------------|------------|------------------|------------------|
| PIDNet       | 0       | -                         | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 30         | 39.74            | 27.41            |
| PIDNet       | 1       | (HF, SSR, RC)             | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 30         | 42.18            | 29.79            |
| DACS         | 0       | (RC-ALL, HF-MXD, SSR-MXD) | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 30         | 33.78            | 23.95            |
| DACS_GCW_LDQ | 0       | (RC-ALL, HF-MXD, SSR-MXD) | Urban      | 6          | CrossEntropyLoss | SGD(momentum: 0.9 weight_decay: 0.0005)  | PolynomialLR(lr=0.001, power=0.9)  | 30         | 33.52            | 19.88            |





## Legend

Data augmentation:
- HF: Horizontal Flip
- SSR: Shift Scale Rotate
- GD: Grid Distortion
- RC: Random Crop
- BC: Brightness Contrast
- CD: Coarse Dropout
- CJ: Color Jitter
- GB: Gaussian Blur

Frequency:
- ALL: All source and mixed data
- SRC: Source data
- MXD: Mixed data
