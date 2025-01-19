import argparse
import os
import re

import random

import numpy as np

import time

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from torch.backends import cudnn
from torch.amp import GradScaler, autocast

from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast, CoarseDropout, GridDistortion, GaussianBlur, ColorJitter, RandomCrop
from albumentations.pytorch import ToTensorV2

from fvcore.nn import FlopCountAnalysis, flop_count_table

from datasets import LoveDADataset, LoveDADatasetLabel, MEAN, STD, NUM_CLASSES, compute_boundaries
from models import PIDNet_S, PIDNet_M, PIDNet_L, FCDiscriminator
from criteria import CrossEntropyLoss, OhemCrossEntropyLoss, BoundaryLoss
from utils import *




def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR10 Classification")

    domains_choices = [
        "Rural",
        "Urban"
    ]

    models_choices = [
        "PIDNet_S",
        "PIDNet_M",
        "PIDNet_L",
    ]

    store_choices = [
        "local",
        "drive",
    ]



    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training mode."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable testing mode."
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Enable predict mode."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from specified model."
    )

    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=1,
        help=f"Specify the epoch to resume.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        choices=models_choices,
        required=True,
        help=f"Specify the model name.",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="0",
        help=f"Specify the version.",
    )

    parser.add_argument(
        "--model_file",
        type=str,
        default="best.pt",
        help=f"Specify the model file name containing weights for testing and prediction.",
    )

    parser.add_argument(
        "--target_domain",
        type=str,
        choices=domains_choices,
        default="Rural",
        help=f"Specify the target domain for testing.",
    )

    parser.add_argument(
        "--horizontal_flip_augmentation",
        action="store_true",
        help="Performs horizontal flip data augmentation on dataset."
    )

    parser.add_argument(
        "--shift_scale_rotate_augmentation",
        action="store_true",
        help="Performs shift scale rotate data augmentation on dataset."
    )

    parser.add_argument(
        "--brightness_contrast_augmentation",
        action="store_true",
        help="Performs random brightness contrast data augmentation on dataset."
    )

    parser.add_argument(
        "--coarse_dropout_augmentation",
        action="store_true",
        help="Performs coarse dropout data augmentation on dataset."
    )

    parser.add_argument(
        "--grid_distortion_augmentation",
        action="store_true",
        help="Performs grid distortion data augmentation on dataset."
    )

    parser.add_argument(
        "--color_jitter_augmentation",
        action="store_true",
        help="Performs color jitter data augmentation on dataset."
    )

    parser.add_argument(
        "--gaussian_blur_augmentation",
        action="store_true",
        help="Performs gaussian blur data augmentation on dataset."
    )

    parser.add_argument(
        "--random_crop_augmentation",
        action="store_true",
        help="Performs random crop data augmentation on dataset."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help=f"Specify the batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help=f"Specify the initial learning rate.",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help=f"Specify the momentum for SGD optimizer.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help=f"Specify the weight decay for SGD optimizer.",
    )

    parser.add_argument(
        "--power",
        type=float,
        default=0.9,
        help=f"Specify power for PolynomialLR scheduler.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help=f"Specify maximum alpha for ema model.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help=f"Specify the number of training epochs.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=1000,
        help=f"Specify the number of epochs necessary for early stopping if there isn't improvement.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help=f"Specify the random seed.",
    )

    parser.add_argument(
        "--store",
        type=str,
        choices=store_choices,
        default="local",
        help=f"Specify where to store results (local, drive).",
    )

    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_results_dir(store, model_name, version, resume):
    if store == "drive":
        res_dir = "/content/drive/MyDrive/res"
    else:
        res_dir = "res"

    os.makedirs(res_dir, exist_ok=True)

    dir_name = f"{model_name}_DACS_{version}"
    if not resume:
        for file in os.listdir(res_dir):
            if file == dir_name:
                raise Exception(f"Directory {dir_name} already exists")

    res_dir = f"{res_dir}/{dir_name}"
    if not resume:
        sub_dirs = [res_dir, f"{res_dir}/weights", f"{res_dir}/plots"]
        for sub_dir in sub_dirs:
            os.makedirs(sub_dir, exist_ok=True)

    return res_dir

def get_results_dir(store, model_name, version):
    if store == "drive":
        res_dir = "/content/drive/MyDrive/res"
    else:
        res_dir = "res"

    dir_name = f"{model_name}_DACS_{version}"
    res_dir = f"{res_dir}/{dir_name}"

    return res_dir


def get_model_number(res_dir):
    model_number = 0

    model_found = False
    pattern = r'last_(\d+)\.pt'
    for file in os.listdir(f"{res_dir}/weights"):
        match = re.match(pattern, file)
        if match:
            model_found = True
            n = int(match.group(1))
            if n > model_number:
                model_number = n

    if model_found:
        model_number += 1
    
    return model_number


def get_device():
    if torch.cuda.is_available():
        print("CUDA available")
        print(f"Number of devices: {torch.cuda.device_count()}")
        for dev in range(torch.cuda.device_count()):
            print(f"Device {dev}:")
            print(f"\tName: {torch.cuda.get_device_name(dev)}")
    else:
        print("CUDA not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    return device


















def log_training_setup(device, args, monitor):
    monitor.log(f"Model: {args.model_name} DACS")

    monitor.log(f"Device: {device}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        monitor.log(f"Cuda device name: {device_name}")

    data_augmentation = args.horizontal_flip_augmentation or args.shift_scale_rotate_augmentation or args.brightness_contrast_augmentation or args.coarse_dropout_augmentation or args.grid_distortion_augmentation or args.color_jitter_augmentation or args.gaussian_blur_augmentation

    monitor.log(f"Data augmentation: {data_augmentation}")
    if args.horizontal_flip_augmentation:
        monitor.log("- HorizontalFlip(p=0.5)")
    if args.shift_scale_rotate_augmentation:
        monitor.log("- ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5)")
    if args.brightness_contrast_augmentation:
        monitor.log("- RandomBrightnessContrast(p=0.5)")
    if args.coarse_dropout_augmentation:
        monitor.log("- CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5)")
    if args.grid_distortion_augmentation:
        monitor.log("- GridDistortion(num_steps=5, distort_limit=0.3, p=0.5)")
    if args.color_jitter_augmentation:
        monitor.log("- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)")
    if args.gaussian_blur_augmentation:
        monitor.log("- GaussianBlur(blur_limit=(3, 7), p=0.5)")
    if args.random_crop_augmentation:
        monitor.log("- RandomCrop(width=720, height=720, p=0.5)")

    monitor.log(f"Batch size: {args.batch_size}\n")



def log_testing_setup(device, args, monitor):
    monitor.log(f"Model test file: {args.model_file}")
    monitor.log(f"Device: {device}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        monitor.log(f"Cuda device name: {device_name}")
    monitor.log(f"Dataset target domain: {args.target_domain}\n")



def dataset_preprocessing(domain, batch_size):
    
    transform = Compose([
        # Resize(512, 512),
        Normalize(mean=MEAN, std=STD, always_apply=True),
        ToTensorV2()
    ])

    # Define the Dataset object for training, validation and testing
    traindataset = LoveDADataset(dataset_type="Train", domain=domain, transform=transform, root_dir='data')
    valdataset = LoveDADataset(dataset_type="Val", domain=domain, transform=transform, root_dir='data')
    testdataset = LoveDADataset(dataset_type="Test", domain=domain, transform=transform, root_dir='data')

    # Define the DataLoaders
    trainloader = DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, num_workers=2)
    testloader = DataLoader(testdataset, batch_size=batch_size, num_workers=2)

    return trainloader, valloader, testloader

def get_transform(args):
    transform_list = []

    if args.horizontal_flip_augmentation:
        transform_list.append(HorizontalFlip(p=0.5))
    if args.shift_scale_rotate_augmentation:
        transform_list.append(ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5))
    if args.brightness_contrast_augmentation:
        transform_list.append(RandomBrightnessContrast(p=0.5))
    if args.coarse_dropout_augmentation:
        transform_list.append(CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5))
    if args.grid_distortion_augmentation:
        transform_list.append(GridDistortion(num_steps=5, distort_limit=0.3, p=0.5))
    if args.color_jitter_augmentation:
        transform_list.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5))
    if args.gaussian_blur_augmentation:
        transform_list.append(GaussianBlur(blur_limit=(3, 7), p=0.5))
    if args.random_crop_augmentation:
        transform_list.append(RandomCrop(width=720, height=720, p=0.5))

    transform_list.append(ToTensorV2())

    transform = Compose(transform_list)

    return transform

def get_model(args, device):
    if args.model_name == "PIDNet_S":
        if args.train:
            model = PIDNet_S(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_s_pretrained_imagenet.pth")
        else:
            model = PIDNet_S(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_s_pretrained_imagenet.pth", augment=False)
        ema_model = PIDNet_S(num_classes=NUM_CLASSES, pretrain=False)
    elif args.model_name == "PIDNet_M":
        if args.train:
            model = PIDNet_M(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_m_pretrained_imagenet.pth")
        else:
            model = PIDNet_M(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_m_pretrained_imagenet.pth", augment=False)
        ema_model = PIDNet_M(num_classes=NUM_CLASSES, pretrain=False)
    elif args.model_name == "PIDNet_L":
        if args.train:
            model = PIDNet_L(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_l_pretrained_imagenet.pth")
        else:
            model = PIDNet_L(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_l_pretrained_imagenet.pth", augment=False)
        ema_model = PIDNet_L(num_classes=NUM_CLASSES, pretrain=False)
    else:
        raise Exception(f"Model {args.model_name} doesn't exist")
    
    for param in ema_model.parameters():
        param.detach_()
        
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()

    model = model.to(device)
    ema_model = ema_model.to(device)
    
    return model, ema_model


def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


def load_model(model, file_name, device):
    model.load_state_dict(torch.load(file_name, map_location=torch.device(device), weights_only=True))
    return model


def get_criterion():
    criterion = CrossEntropyLoss(ignore_label=255)
    bd_criterion = BoundaryLoss()

    return criterion, bd_criterion



def get_optimizer(model, args):
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    return optimizer 


def get_scheduler(optimizer, args):
    max_iters = args.epochs
    power = args.power

    def polynomial_lr(current_iter):
        return (1 - current_iter / max_iters)**power
        
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=polynomial_lr
    )
    
    return scheduler


def compute_mIoU(predictions, masks, num_classes):
    class_iou = torch.zeros(num_classes, dtype=torch.float32)

    predictions = predictions.view(-1)
    masks = masks.view(-1)

    for cls in range(num_classes):
        intersection = torch.sum((predictions == cls) & (masks == cls))
        union = torch.sum((predictions == cls) | (masks == cls))

        if union == 0:
            class_iou[cls] = float('nan')
        else:
            class_iou[cls] = intersection / union

    mean_iou = torch.nanmean(class_iou).item()

    return mean_iou, class_iou








def train(model, ema_model, model_number, src_trainloader, trg_trainloader, src_valloader, trg_valloader, transform, criterion, bd_criterion, \
           optimizer, scheduler, epochs, init_epoch, patience, device, monitor, res_dir, args):
    
    cudnn.benchmark = True

    train_num_steps = min(len(src_trainloader), len(trg_trainloader))
    val_num_steps = min(len(src_valloader), len(trg_valloader))

    train_losses_labeled = []
    train_losses_unlabeled = []
    val_losses = []

    train_mIoUs = []
    val_mIoUs = []

    learning_rates = []

    best_val_loss = None
    best_val_mIoU = None
    patience_counter = 0

    for e in range(init_epoch-1, epochs):
        # Training
        monitor.start(desc=f"Epoch {e + 1}/{epochs}", max_progress=train_num_steps)

        learning_rate = scheduler.get_last_lr()[0]
        learning_rates.append(learning_rate)

        cumulative_loss_labeled = 0.0
        cumulative_loss_unlabeled = 0.0
        cumulative_mIoU = 0.0
        count = 0
        train_mIoU = 0.0

        model.train()
        ema_model.train()

        src_train_iter = iter(src_trainloader)
        trg_train_iter = iter(trg_trainloader)

        for i in range(train_num_steps):
            src_images, src_masks, src_boundaries = next(src_train_iter)
            src_images, src_masks, src_boundaries = src_images.to(device), src_masks.to(device), src_boundaries.to(device)
            trg_images, _, _ = next(trg_train_iter)
            trg_images = trg_images.to(device)

            optimizer.zero_grad()



            # Train Segmentation Network with Labeled data
            # src_transformed_images = []
            # src_transformed_masks = []
            # for j in range(src_images.shape[0]):
            #     image = src_images[j]
            #     mask = src_masks[j]

            #     image = image.squeeze(0).cpu().numpy().transpose((1, 2, 0))             
            #     mask = mask.squeeze(0).cpu().numpy()

            #     transformation = transform(image=image, mask=mask)
            #     image, mask = transformation['image'], transformation['mask']
            #     mask = mask.long()

            #     src_transformed_images.append(image)
            #     src_transformed_masks.append(mask)

            # src_transformed_images = torch.stack(src_transformed_images).to(device)
            # src_transformed_masks = torch.stack(src_transformed_masks).to(device)
            
            
            src_logits = model(src_images)


            h, w = src_masks.size(1), src_masks.size(2)
            ph, pw = src_logits[0].size(2), src_logits[0].size(3)
            if ph != h or pw != w:
                for j in range(len(src_logits)):
                    src_logits[j] = F.interpolate(src_logits[j], size=(h, w), mode='bilinear', align_corners=False)


            loss_s = criterion(src_logits[:-1], src_masks, balance_weights=[0.4, 1.0])
            loss_b = bd_criterion(src_logits[-1], src_boundaries)

            filler = torch.ones_like(src_masks) * 255
            bd_label = torch.where(F.sigmoid(src_logits[-1][:,0,:,:])>0.8, src_masks, filler)
            loss_sb = criterion(src_logits[-2], bd_label)
            
            loss_labeled = loss_s + loss_b + loss_sb

            cumulative_loss_labeled += loss_labeled.item()



            # Train Segmentation Network with Unlabeled data

            trg_logits = ema_model(trg_images)

            h, w = src_masks.size(1), src_masks.size(2)
            ph, pw = trg_logits[0].size(2), trg_logits[0].size(3)
            if ph != h or pw != w:
                for j in range(len(trg_logits)):
                    trg_logits[j] = F.interpolate(trg_logits[j], size=(h, w), mode='bilinear', align_corners=False)

            trg_max_probs, trg_prediction = torch.max(torch.softmax(trg_logits[-2], dim=1), dim=1)


            unlabeled_weight = torch.sum(trg_max_probs.ge(0.968).long() == 1).item() / (np.size(np.array(trg_logits[-2].cpu())) * args.batch_size)
            pixel_wise_weights = unlabeled_weight * torch.ones(trg_max_probs.shape).to(device)
            ones_weights = torch.ones((pixel_wise_weights.shape)).to(device)


            mixed_images = []
            mixed_masks = []
            weights = []
            
            for j in range(src_images.shape[0]):
                classes = torch.unique(src_masks[j])
                n_classes = classes.shape[0]
                classes = (classes[torch.Tensor(np.random.choice(n_classes, int((n_classes + n_classes%2)/2),replace=False)).long()]).to(device)
                filter_mask = generate_class_mask(src_masks[j], classes).unsqueeze(0).to(device)

                image, mask = oneMix(
                    mask=filter_mask,
                    data=torch.cat((src_images[j].unsqueeze(0), trg_images[j].unsqueeze(0))),
                    target=torch.cat((src_masks[j].unsqueeze(0), trg_prediction[j].unsqueeze(0)))
                )

                _, weight = oneMix(
                    mask=filter_mask,
                    target=torch.cat((ones_weights[j].unsqueeze(0), pixel_wise_weights[j].unsqueeze(0)))
                )

                image = image.squeeze(0).cpu().numpy().transpose((1, 2, 0))             
                mask = mask.squeeze(0).cpu().numpy()
                weight = weight.squeeze(0).cpu().numpy()

                tmp_mask = mask

                transformation = transform(image=image, masks=[mask, weight])
                image = transformation['image']
                mask, weight = transformation['masks']

                mask = mask.long()

                mixed_images.append(image)
                mixed_masks.append(mask)

                weights.append(weight)


            mixed_images = torch.stack(mixed_images)
            mixed_masks = torch.stack(mixed_masks)
            weights = torch.stack(weights)

            mixed_boundaries = compute_boundaries(mixed_masks.cpu())
            
            # for image, mask, boundary in zip(mixed_images, mixed_masks, mixed_boundaries):
            #     plot_dataset_entry(
            #         image.numpy(),
            #         mask.numpy(),
            #         boundary.numpy(),
            #         np_format=True,
            #         alpha=1.,
            #         title="Mixed produced data",
            #         show=True
            #     )

            mixed_images = mixed_images.to(device)
            mixed_masks = mixed_masks.to(device)
            mixed_boundaries = mixed_boundaries.to(device)
            weights = weights.to(device)


            mixed_logits = model(mixed_images)

            h, w = src_masks.size(1), src_masks.size(2)
            ph, pw = mixed_logits[0].size(2), mixed_logits[0].size(3)
            if ph != h or pw != w:
                for j in range(len(mixed_logits)):
                    mixed_logits[j] = F.interpolate(mixed_logits[j], size=(h, w), mode='bilinear', align_corners=False)




            loss_s = criterion(mixed_logits[:-1], mixed_masks, balance_weights=[0.4, 1.0], pixel_wise_weights=weights)
            loss_b = bd_criterion(mixed_logits[-1], mixed_boundaries)

            filler = torch.ones_like(mixed_masks) * 255
            bd_label = torch.where(F.sigmoid(mixed_logits[-1][:,0,:,:])>0.8, mixed_masks, filler)
            loss_sb = criterion(mixed_logits[-2], bd_label, pixel_wise_weights=weights)
            
            loss_unlabeled = loss_s + loss_b + loss_sb

            cumulative_loss_unlabeled += loss_unlabeled.item()
            

            loss = loss_labeled + loss_unlabeled
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


            # Update ema_model
            iteration = (train_num_steps * e + count)
            alpha_max = args.alpha
            alpha_teacher = min(1 - 1 / (iteration + 1), alpha_max)
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]



            predictions = torch.argmax(torch.softmax(src_logits[-2], dim=1), dim=1)
            
            count += 1

            train_loss = (cumulative_loss_labeled + cumulative_loss_unlabeled) / count
            train_loss_labeled = cumulative_loss_labeled / count
            train_loss_unlabeled = cumulative_loss_unlabeled / count

            mIoU, _ = compute_mIoU(predictions, src_masks, NUM_CLASSES)
            cumulative_mIoU += mIoU
            train_mIoU = cumulative_mIoU / count

            monitor.update(
                i + 1,
                learning_rate=f"{learning_rate:.5f}",
                train_loss=f"{train_loss:.4f}",
                train_loss_labeled=f"{train_loss_labeled:.4f}",
                train_loss_unlabeled=f"{train_loss_unlabeled:.4f}",
                train_mIoU=f"{train_mIoU:.4f}",
            )

        train_losses_labeled.append(train_loss_labeled)
        train_losses_unlabeled.append(train_loss_unlabeled)
        train_losses = [a + b for a, b in zip(train_losses_labeled, train_losses_unlabeled)]
        train_mIoUs.append(train_mIoU)

        monitor.stop()










        # Validation
        monitor.start(desc=f"Validation", max_progress=val_num_steps)

        cumulative_loss = 0.0
        cumulative_mIoU = 0.0
        count = 0
        val_mIoU = 0.0
        val_loss = 0.0

        model.eval()
        ema_model.eval()

        src_val_iter = iter(src_valloader)

        with torch.no_grad():
            for i in range(val_num_steps):
                src_images, src_masks, src_boundaries = next(src_val_iter)
                src_images, src_masks, src_boundaries = src_images.to(device), src_masks.to(device), src_boundaries.to(device)

                src_logits = model(src_images)

                h, w = src_masks.size(1), src_masks.size(2)
                ph, pw = src_logits[0].size(2), src_logits[0].size(3)
                if ph != h or pw != w:
                    for j in range(len(src_logits)):
                        src_logits[j] = F.interpolate(src_logits[j], size=(h, w), mode='bilinear', align_corners=False)


                loss_s = criterion(src_logits[:-1], src_masks, balance_weights=[0.4, 1.0])
                loss_b = bd_criterion(src_logits[-1], src_boundaries)

                filler = torch.ones_like(src_masks) * 255
                bd_label = torch.where(F.sigmoid(src_logits[-1][:,0,:,:])>0.8, src_masks, filler)
                loss_sb = criterion(src_logits[-2], bd_label)
                
                loss = loss_s + loss_b + loss_sb

                cumulative_loss += loss.item()

                    
                predictions = torch.argmax(torch.softmax(src_logits[-2], dim=1), dim=1)
            
                count += 1

                val_loss = cumulative_loss / count

                mIoU, _ = compute_mIoU(predictions, src_masks, NUM_CLASSES)
                cumulative_mIoU += mIoU
                val_mIoU = cumulative_mIoU / count

                monitor.update(
                    i + 1,
                    learning_rate=f"{learning_rate:.5f}",
                    val_loss=f"{val_loss:.4f}",
                    val_mIoU=f"{val_mIoU:.4f}",
                )


        val_losses.append(val_loss)
        val_mIoUs.append(val_mIoU)

        monitor.stop()


        if (best_val_loss is None and best_val_mIoU is None) or val_loss < best_val_loss or val_mIoU > best_val_mIoU:
            save_model(model, f"{res_dir}/weights/best_{model_number}.pt")
            monitor.log(f"Model saved as best_{model_number}.pt\n")
            best_val_loss = val_loss
            best_val_mIoU = val_mIoU
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            monitor.log(f"Early stopping after {e + 1} epochs\n")
            break


        scheduler.step()

        save_model(model, f"{res_dir}/weights/last_{model_number}.pt")
        save_model(ema_model, f"{res_dir}/weights/last_ema_{model_number}.pt")

        plot_metrics(
            values_list=[train_losses],
            labels=["Train Cumulative Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            res_dir=res_dir,
            file_name=f"loss_{model_number}"
        )

        plot_metrics(
            values_list=[train_losses_labeled, val_losses],
            labels=["Train Labeled Loss", "Val Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            res_dir=res_dir,
            file_name=f"loss_seg_{model_number}"
        )

        plot_metrics(
            values_list=[train_losses_labeled, train_losses_unlabeled],
            labels=["Train Labeled Loss", "Train Unlabeled Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            res_dir=res_dir,
            file_name=f"loss_train_{model_number}"
        )

        plot_metrics(
            values_list=[train_mIoUs, val_mIoUs],
            labels=["Train mIoU", "Val mIoU"],
            title="Mean Intersection over Union",
            xlabel="Epoch",
            ylabel="mIoU",
            res_dir=res_dir,
            file_name=f"mIoU_{model_number}"
        )

        plot_metrics(
            values_list=[learning_rates],
            labels=["Learning rate"],
            title="Learning rate",
            xlabel="Epoch",
            ylabel="lr",
            res_dir=res_dir,
            file_name=f"learning_rate_{model_number}"
        )

    monitor.print_stats()











def test(model, valloader, device, monitor):
    monitor.start(desc=f"Testing", max_progress=len(valloader))

    flops_count = 0
    cumulative_mIoU = 0.0
    count = 0
    test_mIoU = 0.0
    inference_times = []

    cumulative_class_iou = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    class_count = torch.zeros(NUM_CLASSES, dtype=torch.int32)

    model.eval()
    with torch.no_grad():
        
        # FLOPs analysis
        images, _, _ = next(iter(valloader))
        images = images.to(device)
        flops = FlopCountAnalysis(model, images)
        flops_count = flop_count_table(flops)

        # Testing
        for i, (images, masks, _) in enumerate(valloader):
            images, masks = images.to(device), masks.to(device)

            start_time = time.perf_counter()
            logits = model(images)
            end_time = time.perf_counter()

            h, w = masks.size(1), masks.size(2)
            ph, pw = logits.size(1), logits.size(2)
            if ph != h or pw != w:
                logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)


            batch_inference_time = (end_time - start_time) / images.size(0)
            inference_times.append(batch_inference_time)

            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            count += 1

            mIoU, class_iou = compute_mIoU(predictions, masks, NUM_CLASSES)

            valid_class_mask = ~torch.isnan(class_iou)
            cumulative_class_iou[valid_class_mask] += class_iou[valid_class_mask]
            class_count[valid_class_mask] += 1

            
            cumulative_mIoU += mIoU
            test_mIoU = cumulative_mIoU / count
            
            monitor.update(
                i + 1,
                test_mIoU=f"{test_mIoU:.4f}",
            )

    monitor.stop()

    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    final_class_iou = cumulative_class_iou / class_count.clamp(min=1)

    monitor.log(f"Model parameters and FLOPs:\n{flops_count}\n")
    monitor.log(f"Mean Intersection over Union on test images: {test_mIoU*100:.3f} %")
    for label in LoveDADatasetLabel:
        monitor.log(f"\t{label.name} IoU: {final_class_iou[label.value]*100:.3f} %")
    monitor.log(f"")
    monitor.log(f"Mean inference time: {mean_inference_time * 1000:.3f} ms")
    monitor.log(f"Standard deviation of inference time: {std_inference_time * 1000:.3f} ms")







def predict(model, valloader, device):
    model.eval()
    with torch.no_grad():

        # Predicting
        for i, (images, masks, _) in enumerate(valloader):
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            h, w = masks.size(1), masks.size(2)
            ph, pw = logits.size(1), logits.size(2)
            if ph != h or pw != w:
                logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)

            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            mIoU, iou_per_class = compute_mIoU(predictions, masks, NUM_CLASSES)

            iou_per_class_str = ""
            for label in LoveDADatasetLabel:
                iou_per_class_str += f"{label.name}: {iou_per_class[label.value]*100:.3f} %\n"            

            plot_prediction(images[0], masks[0], predictions[0], alpha=0.4, title="Prediction", description=f"Mean Intersection over Union: {mIoU*100:.3f} %\n\n{iou_per_class_str}")










def main():
    args = parse_args()

    if args.train + args.test + args.predict > 1:
        raise Exception("Only one task allowed, selected more than one (train, test, predict)")

    set_seed(args.seed)
    device = get_device()

    if args.train:
        res_dir = make_results_dir(args.store, args.model_name, args.version, args.resume)

        file_name = f"{res_dir}/training_log.txt"
        train_monitor = Monitor(file_name, resume=args.resume, inline=False)

        src_trainloader, src_valloader, _ = dataset_preprocessing(
            domain="Urban",
            batch_size=args.batch_size
        )

        trg_trainloader, trg_valloader, _ = dataset_preprocessing(
            domain="Rural",
            batch_size=args.batch_size
        )
        
        # inspect_dataset(src_trainloader, src_valloader)
        
        transform = get_transform(args)

        model, ema_model = get_model(args, device)

        model_number = get_model_number(res_dir)

        if args.resume:
            model = load_model(model, f"{res_dir}/weights/last_{model_number-1}.pt", device)
            ema_model = load_model(ema_model, f"{res_dir}/weights/last_ema_{model_number-1}.pt", device)


        criterion, bd_criterion = get_criterion()
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        if args.resume:
            for _ in range(args.resume_epoch-1):
                scheduler.step()

        log_training_setup(device, args, train_monitor)

        train(
            model=model,
            ema_model=ema_model,
            model_number=model_number,
            src_trainloader=src_trainloader,
            trg_trainloader=trg_trainloader,
            src_valloader=src_valloader,
            trg_valloader=trg_valloader,
            transform=transform,
            criterion=criterion,
            bd_criterion=bd_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            init_epoch=args.resume_epoch,
            patience=args.patience,
            device=device,
            monitor=train_monitor,
            res_dir=res_dir,
            args=args
        )
    

    if args.test:
        res_dir = get_results_dir(args.store, args.model_name, args.version)

        file_name = f"{res_dir}/testing_log.txt"
        resume = os.path.exists(file_name)
        test_monitor = Monitor(file_name, resume, inline=False)

        trainloader, valloader, _ = dataset_preprocessing(
            domain=args.target_domain,
            batch_size=args.batch_size
        )

        model, _ = get_model(args, device)
        model = load_model(model, f"{res_dir}/weights/{args.model_file}", device)

        log_testing_setup(device, args, test_monitor)

        test(
            model=model,
            valloader=valloader,
            device=device,
            monitor=test_monitor
        )

    
    if args.predict:
        res_dir = get_results_dir(args.store, args.model_name, args.version)

        trainloader, valloader, _ = dataset_preprocessing(
            domain=args.target_domain,
            batch_size=1
        )

        model, _ = get_model(args, device)
        model = load_model(model, f"{res_dir}/weights/{args.model_file}", device)

        predict(
            model=model,
            valloader=valloader,
            device=device
        )



if __name__ == "__main__":
    main()
