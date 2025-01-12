# Train DeepLabV2_ResNet101
# python3 main.py \
#     --train \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --source_domain Rural \
#     --batch_size 8 \
#     --optimizer SGD \
#     --scheduler PolynomialLR \
#     --lr 0.01 \
#     --power 0.6 \
#     --epochs 20


# Resume DeepLabV2_ResNet101
# python3 main.py \
#     --train \
#     --resume \
#     --resume_epoch 14 \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --source_domain Rural \
#     --batch_size 8 \
#     --optimizer SGD \
#     --scheduler PolynomialLR \
#     --lr 0.01 \
#     --power 0.6 \
#     --epochs 20


# Test DeepLabV2_ResNet101 last_0.pt
# python3 main.py \
#     --test \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --test_model_file last_0.pt \
#     --target_domain Rural \
#     --batch_size 8


# Test DeepLabV2_ResNet101 best_0.pt
# python3 main.py \
#     --test \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --test_model_file last_0.pt \
#     --target_domain Rural \
#     --batch_size 8









# Train PIDNet_S
python3 main.py \
    --train \
    --model_name PIDNet_S \
    --version 2 \
    --source_domain Rural \
    --data_augmentation \
    --batch_size 8 \
    --optimizer SGD \
    --weight_decay 0.001 \
    --scheduler PolynomialLR \
    --lr 0.01 \
    --power 0.9 \
    --epochs 20


# Test PIDNet_S last.pt
# python3 main.py \
#     --test \
#     --model_name PIDNet_S \
#     --version 1 \
#     --test_model_file last_0.pt \
#     --target_domain Rural \
#     --batch_size 8


# Test PIDNet_S best.pt
# python3 main.py \
#     --test \
#     --model_name PIDNet_S \
#     --version 1 \
#     --test_model_file best_0.pt \
#     --target_domain Rural \
#     --batch_size 8