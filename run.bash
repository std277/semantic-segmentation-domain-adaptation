# Train
python3 main.py \
    --train \
    --source_domain Rural \
    --model_name DeepLabV2_ResNet101 \
    --version 0 \
    --batch_size 4 \
    --optimizer SGD \
    --scheduler PolynomialLR \
    --lr 0.01 \
    --power 0.6 \
    --epochs 20

# Test
# python3 main.py \
#     --test \
#     --target_domain Rural \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --test_model_file best.pt