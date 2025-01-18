# semantic-segmentation-domain-adaptation
Real-time Domain Adaptation in Semantic Segmentation

## Trainings

Model name: `DeepLabV2_ResNet101`

Resizing: (512, 512)

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                            | SCHEDULER                        | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------------------|----------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 20         | 17.85            | 22.42            |


Model parameters: 43.016M

FLOPs: 1.11T

Mean inference time: 4.032 ms

Standard deviation of inference time: 3.590 ms

(Latencies computed on google colab with Tesla T4 GPU)






Model name: `PIDNet_S`

Resizing: (512, 512)

| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                            | SCHEDULER                        | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------------------|----------------------------------|------------|------------------|------------------|
| 0       | -                  | Rural      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 20         | 36.84            | 25.25            |
| 1       | -                  | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 20         | 37.89            | 24.13            |
| 2       | (HF, SSR)          | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 30         | 36.96            | 27.21            |
| 3       | (BC, CD)           | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 30         | 35.36            | 23.13            |
| 4       | (HF, SSR, BC, CD)  | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 30         | 33.14            | 19.21            |
| 5       | (HF, SSR, GD)      | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 30         | 38.52            | 27.45            |

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

OLD:
| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                            | SCHEDULER                        | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------------------|----------------------------------|------------|------------------|------------------|
| 0       | -                  | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 30         | 33.31            | 18.67            |
| 1       | (CJ, GB)           | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 30         | 30.35            | 20.34            | 
| 2       | (CJ, GB)           | Urban      | 2          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 30         | 29.24            | 19.36            | 
| 3       | (CJ, GB)           | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.001, momentum: 0.9 weight_decay: 0.0005)   | PolynomialLR(lr=0.01, power=0.9) | 30         | 33.33            | 20.59            | 
| 4       | (CJ, GB)           | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.00025, momentum: 0.9 weight_decay: 0.0005) | PolynomialLR(lr=0.01, power=0.9) | 30         | 31.00            | 20.43            | 
| 5       | (CJ, GB)           | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.001, momentum: 0.9 weight_decay: 0.0005)   | PolynomialLR(lr=0.01, power=0.9) | 30         | 32.26            | 19.63            | (alpha=0.9)


| VERSION | DATA AUG           | SRC DOMAIN | BATCH SIZE | CRITERION            | OPTIMIZER                                            | SCHEDULER                        | NUM_EPOCHS | mIoU (%) (Urban) | mIoU (%) (Rural) |
|---------|--------------------|------------|------------|----------------------|------------------------------------------------------|----------------------------------|------------|------------------|------------------|
| 0       | -                  | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 30         | 33.31            | 18.67            |
| 1       | (CJ, GB)           | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.01, momentum: 0.9 weight_decay: 0.0005)    | PolynomialLR(lr=0.01, power=0.9) | 30         | 30.35            | 20.34            | 
| 2       | (CJ, GB)           | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.001, momentum: 0.9 weight_decay: 0.0005)   | PolynomialLR(lr=0.01, power=0.9) | 30         | 33.33            | 20.59            | 
| 3       | (HF, SSR, GD)      | Urban      | 6          | CrossEntropyLoss     | SGD(lr: 0.001, momentum: 0.9 weight_decay: 0.0005)   | PolynomialLR(lr=0.01, power=0.9) | 30         |             |             | 


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



Optional:
- Fix OhemCrossEntropyLoss
