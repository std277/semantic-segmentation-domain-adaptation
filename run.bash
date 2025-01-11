# Train
# python3 main.py \
#     --train \
#     --model_name DeepLabV2_ResNet101 \
#     --version 0 \
#     --source_domain Rural \
#     --batch_size 2 \
#     --optimizer SGD \
#     --scheduler PolynomialLR \
#     --lr 0.01 \
#     --power 0.6 \
#     --epochs 20

# Test
python3 main.py \
    --test \
    --model_name DeepLabV2_ResNet101 \
    --version 0 \
    --test_model_file best.pt \
    --target_domain Rural \
    --batch_size 2