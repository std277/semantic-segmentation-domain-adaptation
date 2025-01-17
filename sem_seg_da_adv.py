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

from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast, CoarseDropout, GridDistortion
from albumentations.pytorch import ToTensorV2

from fvcore.nn import FlopCountAnalysis, flop_count_table

from datasets import LoveDADataset, LoveDADatasetLabel, MEAN, STD, NUM_CLASSES
from models import PIDNet_S, PIDNet_M, PIDNet_L, FCDiscriminator
from criteria import CrossEntropyLoss, OhemCrossEntropyLoss, BoundaryLoss
from utils import *




def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR10 Classification")

    mode_choices = [
        "single_level",
        "multi_level"
    ]

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
        "--mode",
        type=str,
        choices=mode_choices,
        default="single_level",
        help=f"Specify the number of adversarial network.",
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
        "--batch_size",
        type=int,
        default=8,
        help=f"Specify the batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help=f"Specify the initial learning rate.",
    )

    parser.add_argument(
        "--lr_D",
        type=float,
        default=0.01,
        help=f"Specify the initial learning rate for discriminator.",
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

    dir_name = f"{model_name}_Adversarial_{version}"
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

    dir_name = f"{model_name}_Adversarial_{version}"
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
    monitor.log(f"Model: {args.model_name} Adversarial Discriminator")
    monitor.log(f"Mode: {args.mode}\n")

    monitor.log(f"Device: {device}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        monitor.log(f"Cuda device name: {device_name}")

    data_augmentation = args.horizontal_flip_augmentation or args.shift_scale_rotate_augmentation or args.brightness_contrast_augmentation or args.coarse_dropout_augmentation or args.grid_distortion_augmentation
    monitor.log(f"Data augmentation: {data_augmentation}")
    if args.horizontal_flip_augmentation:
        monitor.log(f"- HorizontalFlip(p=0.5)")
    if args.shift_scale_rotate_augmentation:
        monitor.log(f"- ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5)")
    if args.brightness_contrast_augmentation:
        monitor.log(f"- RandomBrightnessContrast(p=0.5)")
    if args.coarse_dropout_augmentation:
        monitor.log(f"- CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5)")
    if args.grid_distortion_augmentation:
        monitor.log(f"- GridDistortion(num_steps=5, distort_limit=0.3, p=0.5)")

    monitor.log(f"Batch size: {args.batch_size}\n")



def log_testing_setup(device, args, monitor):
    monitor.log(f"Model test file: {args.model_file}")
    monitor.log(f"Device: {device}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        monitor.log(f"Cuda device name: {device_name}")
    monitor.log(f"Dataset target domain: {args.target_domain}\n")



def dataset_preprocessing(domain, batch_size, data_augmentation, args):
    
    # Define transforms
    if data_augmentation:
        transform_list = []
        transform_list.append(Resize(512, 512))
        transform_list.append(Normalize(mean=MEAN, std=STD))

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

        transform_list.append(ToTensorV2())

        transform = Compose(transform_list)

    else:
        transform = Compose([
            Resize(512, 512),
            Normalize(mean=MEAN, std=STD),
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


def get_model(args, device):
    if args.model_name == "PIDNet_S":
        model = PIDNet_S(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_s_pretrained_imagenet.pth")
    elif args.model_name == "PIDNet_M":
        model = PIDNet_M(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_m_pretrained_imagenet.pth")
    elif args.model_name == "PIDNet_L":
        model = PIDNet_L(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_l_pretrained_imagenet.pth")
    else:
        raise Exception(f"Model {args.model_name} doesn't exist")
    model = model.to(device)

    model_D2 = FCDiscriminator(num_classes=NUM_CLASSES)
    model_D2 = model_D2.to(device)

    if args.mode == "multi_level":
        model_D1 = FCDiscriminator(num_classes=NUM_CLASSES)
        model_D1 = model_D1.to(device)

        return model, model_D1, model_D2
    
    return model, None, model_D2


def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


def load_model(model, file_name, device):
    model.load_state_dict(torch.load(file_name, map_location=torch.device(device), weights_only=True))
    return model


def get_criterion():
    criterion = CrossEntropyLoss(ignore_label=255)
    bd_criterion = BoundaryLoss()
    bce_criterion = BCEWithLogitsLoss()

    return criterion, bd_criterion, bce_criterion



def get_optimizer(model, model_D1, model_D2, args):
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    optimizer_D2 = Adam(
        model_D2.parameters(),
        lr=args.lr_D,
        betas=(0.9, 0.99)
    )

    if args.mode == "multi_level":
        optimizer_D1 = Adam(
            model_D1.parameters(),
            lr=args.lr_D,
            betas=(0.9, 0.99)
        )
        return optimizer, optimizer_D1, optimizer_D2 
    
    return optimizer, None, optimizer_D2 


def get_scheduler(optimizer, optimizer_D1, optimizer_D2, args):
    max_iters = args.epochs
    power = args.power

    def polynomial_lr(current_iter):
        return (1 - current_iter / max_iters)**power
        
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=polynomial_lr
    )

    scheduler_D2 = LambdaLR(
        optimizer_D2,
        lr_lambda=polynomial_lr
    )

    if args.mode == "multi_level":
        scheduler_D1 = LambdaLR(
            optimizer_D1,
            lr_lambda=polynomial_lr
        )
        return scheduler, scheduler_D1, scheduler_D2
    
    return scheduler, None, scheduler_D2


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























def train_multi_level(model, model_D1, model_D2, model_number, src_trainloader, trg_trainloader, src_valloader, trg_valloader, criterion, bd_criterion, bce_criterion, \
           optimizer, optimizer_D1, optimizer_D2, scheduler, scheduler_D1, scheduler_D2, epochs, init_epoch, patience, device, monitor, res_dir):
    
    cudnn.benchmark = True

    src_label = 0
    trg_label = 1

    train_num_steps = min(len(src_trainloader), len(trg_trainloader))
    val_num_steps = min(len(src_valloader), len(trg_valloader))

    train_seg_losses = []
    train_adv_losses = []
    train_D_losses = []

    val_losses = []

    train_mIoUs = []
    val_mIoUs = []

    learning_rates = []

    best_val_loss = None
    patience_counter = 0

    for e in range(init_epoch-1, epochs):
        # Training
        monitor.start(desc=f"Epoch {e + 1}/{epochs}", max_progress=train_num_steps)

        learning_rate = scheduler.get_last_lr()[0]
        learning_rates.append(learning_rate)

        cumulative_seg_loss = 0.0
        cumulative_adv_loss = 0.0
        cumulative_D_loss = 0.0

        cumulative_mIoU = 0.0
        count = 0
        train_mIoU = 0.0


        model.train()
        model_D1.train()
        model_D2.train()

        src_train_iter = iter(src_trainloader)
        trg_train_iter = iter(trg_trainloader)

        for i in range(train_num_steps):
            src_images, src_masks, src_boundaries = next(src_train_iter)
            src_images, src_masks, src_boundaries = src_images.to(device), src_masks.to(device), src_boundaries.to(device)
            trg_images, _, _ = next(trg_train_iter)
            trg_images = trg_images.to(device)


            optimizer.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()


            src_logits = model(src_images)

            h, w = src_masks.size(1), src_masks.size(2)
            ph, pw = src_logits[0].size(2), src_logits[0].size(3)
            if ph != h or pw != w:
                for j in range(len(src_logits)):
                    src_logits[j] = F.interpolate(src_logits[j], size=(h, w), mode='bilinear', align_corners=False)

            trg_logits = model(trg_images)

            h, w = src_masks.size(1), src_masks.size(2)
            ph, pw = trg_logits[0].size(2), trg_logits[0].size(3)
            if ph != h or pw != w:
                for j in range(len(trg_logits)):
                    trg_logits[j] = F.interpolate(trg_logits[j], size=(h, w), mode='bilinear', align_corners=False)




            # Train Segmentation Network
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False

            ## Training with Source
            loss_s = criterion(src_logits[:-1], src_masks, balance_weights=[0.4, 1.0])
            loss_b = bd_criterion(src_logits[-1], src_boundaries)

            filler = torch.ones_like(src_masks) * 255
            bd_label = torch.where(F.sigmoid(src_logits[-1][:,0,:,:])>0.8, src_masks, filler)
            loss_sb = criterion(src_logits[-2], bd_label)
            
            loss_seg = loss_s + loss_b + loss_sb
            # loss_seg.backward(retain_graph=True)

            cumulative_seg_loss += loss_seg.item()

            ## Training with Target
            D1_out = model_D1(F.softmax(trg_logits[-3], dim=1))
            D2_out = model_D2(F.softmax(trg_logits[-2], dim=1))
            loss_adv1 = bce_criterion(D1_out, torch.full_like(D1_out, src_label, device=device))
            loss_adv2 = bce_criterion(D2_out, torch.full_like(D2_out, src_label, device=device))
            lambda_adv1 = 0.0002
            lambda_adv2 = 0.001
            loss_adv = loss_adv1 * lambda_adv1 + loss_adv2 * lambda_adv2
            # loss_adv.backward(retain_graph=True)

            cumulative_adv_loss += loss_adv.item()



            # Train Discriminant Network
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True

            ## Training with Source
            D1_out = model_D1(F.softmax(src_logits[-3], dim=1))
            D2_out = model_D2(F.softmax(src_logits[-2], dim=1))
            loss_D1_src = bce_criterion(D1_out, torch.full_like(D1_out, src_label, device=device))
            loss_D2_src = bce_criterion(D2_out, torch.full_like(D2_out, src_label, device=device))
            loss_D_src = (loss_D1_src + loss_D2_src) / 2
            # loss_D_src.backward(retain_graph=True)

            cumulative_D_loss += loss_D1_src.item() + loss_D2_src.item()


            ## Training with Target
            D1_out = model_D1(F.softmax(trg_logits[-3], dim=1))
            D2_out = model_D2(F.softmax(trg_logits[-2], dim=1))
            loss_D1_trg = bce_criterion(D1_out, torch.full_like(D1_out, trg_label, device=device))
            loss_D2_trg = bce_criterion(D2_out, torch.full_like(D2_out, trg_label, device=device))
            loss_D_trg = (loss_D1_trg + loss_D2_trg) / 2
            # loss_D_trg.backward(retain_graph=True)

            cumulative_D_loss += loss_D1_trg.item() + loss_D2_trg.item()



            loss = loss_seg + loss_adv + loss_D_src + loss_D_trg
            loss.backward()



            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_D1.step()
            optimizer_D2.step()

            predictions = torch.argmax(torch.softmax(src_logits[-2], dim=1), dim=1)
            
            count += 1

            train_seg_loss = cumulative_seg_loss / count
            train_adv_loss = cumulative_adv_loss / count
            train_D_loss = cumulative_D_loss / count

            mIoU, _ = compute_mIoU(predictions, src_masks, NUM_CLASSES)
            cumulative_mIoU += mIoU
            train_mIoU = cumulative_mIoU / count

            monitor.update(
                i + 1,
                learning_rate=f"{learning_rate:.5f}",
                train_seg_loss=f"{train_seg_loss:.4f}",
                train_adv_loss=f"{train_adv_loss:.4f}",
                train_D2_loss=f"{train_D_loss:.4f}",
                train_mIoU=f"{train_mIoU:.4f}",
            )

        train_seg_losses.append(train_seg_loss)
        train_adv_losses.append(train_adv_loss)
        train_D_losses.append(train_D_loss)
        train_mIoUs.append(train_mIoU)

        monitor.stop()




        # Validation
        monitor.start(desc=f"Validation", max_progress=val_num_steps)

        cumulative_loss = 0.0
        cumulative_mIoU = 0.0
        count = 0
        val_mIoU = 0.0

        model.eval()
        model_D1.eval()
        model_D2.eval()

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


        if best_val_loss is None or val_loss < best_val_loss:
            save_model(model, f"{res_dir}/weights/best_{model_number}.pt")
            save_model(model_D1, f"{res_dir}/weights/best_D1_{model_number}.pt")
            save_model(model_D2, f"{res_dir}/weights/best_D2_{model_number}.pt")
            monitor.log(f"Model saved as best_{model_number}.pt\n")
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            monitor.log(f"Early stopping after {e + 1} epochs\n")
            break




        scheduler.step()
        scheduler_D1.step()
        scheduler_D2.step()

        save_model(model, f"{res_dir}/weights/last_{model_number}.pt")
        save_model(model_D1, f"{res_dir}/weights/last_D1_{model_number}.pt")
        save_model(model_D2, f"{res_dir}/weights/last_D2_{model_number}.pt")


        plot_metrics(
            values_list=[train_seg_losses],
            labels=["Train Seg Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            res_dir=res_dir,
            file_name=f"loss_seg_{model_number}"
        )

        plot_metrics(
            values_list=[train_adv_losses],
            labels=["Train Adv Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            res_dir=res_dir,
            file_name=f"loss_adv_{model_number}"
        )

        plot_metrics(
            values_list=[train_D_losses],
            labels=["Train D Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            res_dir=res_dir,
            file_name=f"loss_D_{model_number}"
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
















def train_single_level(model, model_D2, model_number, src_trainloader, trg_trainloader, src_valloader, trg_valloader, criterion, bd_criterion, bce_criterion, \
           optimizer, optimizer_D2, scheduler, scheduler_D2, epochs, init_epoch, patience, device, monitor, res_dir):
    
    cudnn.benchmark = True

    src_label = 0
    trg_label = 1

    train_num_steps = min(len(src_trainloader), len(trg_trainloader))
    val_num_steps = min(len(src_valloader), len(trg_valloader))

    train_seg_losses = []
    train_adv_losses = []
    train_D_losses = []

    val_losses = []

    train_mIoUs = []
    val_mIoUs = []

    learning_rates = []

    best_val_loss = None
    patience_counter = 0

    for e in range(init_epoch-1, epochs):
        # Training
        monitor.start(desc=f"Epoch {e + 1}/{epochs}", max_progress=train_num_steps)

        learning_rate = scheduler.get_last_lr()[0]
        learning_rates.append(learning_rate)

        cumulative_seg_loss = 0.0
        cumulative_adv2_loss = 0.0
        cumulative_D2_loss = 0.0

        cumulative_mIoU = 0.0
        count = 0
        train_mIoU = 0.0


        model.train()
        model_D2.train()

        src_train_iter = iter(src_trainloader)
        trg_train_iter = iter(trg_trainloader)

        for i in range(train_num_steps):
            src_images, src_masks, src_boundaries = next(src_train_iter)
            src_images, src_masks, src_boundaries = src_images.to(device), src_masks.to(device), src_boundaries.to(device)
            trg_images, _, _ = next(trg_train_iter)
            trg_images = trg_images.to(device)


            optimizer.zero_grad()
            optimizer_D2.zero_grad()


            src_logits = model(src_images)

            h, w = src_masks.size(1), src_masks.size(2)
            ph, pw = src_logits[0].size(2), src_logits[0].size(3)
            if ph != h or pw != w:
                for j in range(len(src_logits)):
                    src_logits[j] = F.interpolate(src_logits[j], size=(h, w), mode='bilinear', align_corners=False)

            trg_logits = model(trg_images)

            h, w = src_masks.size(1), src_masks.size(2)
            ph, pw = trg_logits[0].size(2), trg_logits[0].size(3)
            if ph != h or pw != w:
                for j in range(len(trg_logits)):
                    trg_logits[j] = F.interpolate(trg_logits[j], size=(h, w), mode='bilinear', align_corners=False)




            # Train Segmentation Network
            for param in model_D2.parameters():
                param.requires_grad = False

            ## Training with Source
            loss_s = criterion(src_logits[:-1], src_masks, balance_weights=[0.4, 1.0])
            loss_b = bd_criterion(src_logits[-1], src_boundaries)

            filler = torch.ones_like(src_masks) * 255
            bd_label = torch.where(F.sigmoid(src_logits[-1][:,0,:,:])>0.8, src_masks, filler)
            loss_sb = criterion(src_logits[-2], bd_label)
            
            loss_seg = loss_s + loss_b + loss_sb
            # loss_seg.backward(retain_graph=True)

            cumulative_seg_loss += loss_seg.item()

            ## Training with Target
            D2_out = model_D2(F.softmax(trg_logits[-2], dim=1))
            loss_adv2 = bce_criterion(D2_out, torch.full_like(D2_out, src_label, device=device))
            lambda_adv2 = 0.001
            loss_adv2 = loss_adv2 * lambda_adv2
            # loss_adv2.backward(retain_graph=True)

            cumulative_adv2_loss += loss_adv2.item()



            # Train Discriminant Network
            for param in model_D2.parameters():
                param.requires_grad = True

            ## Training with Source
            D2_out = model_D2(F.softmax(src_logits[-2], dim=1))
            loss_D2_src = bce_criterion(D2_out, torch.full_like(D2_out, src_label, device=device))
            loss_D2_src = loss_D2_src / 2
            # loss_D2_src.backward(retain_graph=True)

            cumulative_D2_loss += loss_D2_src.item()


            ## Training with Target
            D2_out = model_D2(F.softmax(trg_logits[-2], dim=1))
            loss_D2_trg = bce_criterion(D2_out, torch.full_like(D2_out, trg_label, device=device))
            loss_D2_trg = loss_D2_trg / 2
            # loss_D2_trg.backward()

            cumulative_D2_loss += loss_D2_trg.item()





            
            loss = loss_seg + loss_adv2 + loss_D2_src + loss_D2_trg
            loss.backward()







            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_D2.step()

            predictions = torch.argmax(torch.softmax(src_logits[-2], dim=1), dim=1)
            
            count += 1

            train_seg_loss = cumulative_seg_loss / count
            train_adv2_loss = cumulative_adv2_loss / count
            train_D2_loss = cumulative_D2_loss / count

            mIoU, _ = compute_mIoU(predictions, src_masks, NUM_CLASSES)
            cumulative_mIoU += mIoU
            train_mIoU = cumulative_mIoU / count

            monitor.update(
                i + 1,
                learning_rate=f"{learning_rate:.5f}",
                train_seg_loss=f"{train_seg_loss:.4f}",
                train_adv2_loss=f"{train_adv2_loss:.4f}",
                train_D2_loss=f"{train_D2_loss:.4f}",
                train_mIoU=f"{train_mIoU:.4f}",
            )

        train_seg_losses.append(train_seg_loss)
        train_adv_losses.append(train_adv2_loss)
        train_D_losses.append(train_D2_loss)
        train_mIoUs.append(train_mIoU)

        monitor.stop()




        # Validation
        monitor.start(desc=f"Validation", max_progress=val_num_steps)

        cumulative_loss = 0.0
        cumulative_mIoU = 0.0
        count = 0
        val_mIoU = 0.0

        model.eval()
        model_D2.eval()

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


        if best_val_loss is None or val_loss < best_val_loss:
            save_model(model, f"{res_dir}/weights/best_{model_number}.pt")
            save_model(model_D2, f"{res_dir}/weights/best_D2_{model_number}.pt")
            monitor.log(f"Model saved as best_{model_number}.pt\n")
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            monitor.log(f"Early stopping after {e + 1} epochs\n")
            break


        scheduler.step()
        scheduler_D2.step()

        save_model(model, f"{res_dir}/weights/last_{model_number}.pt")
        save_model(model_D2, f"{res_dir}/weights/last_D2_{model_number}.pt")


        plot_metrics(
            values_list=[train_seg_losses],
            labels=["Train Seg Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            res_dir=res_dir,
            file_name=f"loss_seg_{model_number}"
        )

        plot_metrics(
            values_list=[train_adv_losses],
            labels=["Train Adv Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            res_dir=res_dir,
            file_name=f"loss_adv_{model_number}"
        )

        plot_metrics(
            values_list=[train_D_losses],
            labels=["Train D Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            res_dir=res_dir,
            file_name=f"loss_D_{model_number}"
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











def test(model_name, model, valloader, device, monitor):
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
            ph, pw = logits[0].size(2), logits[0].size(3)
            if ph != h or pw != w:
                for j in range(len(logits)):
                    logits[j] = F.interpolate(logits[j], size=(h, w), mode='bilinear', align_corners=False)


            batch_inference_time = (end_time - start_time) / images.size(0)
            inference_times.append(batch_inference_time)

            predictions = torch.argmax(torch.softmax(logits[-2], dim=1), dim=1)
            
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







def predict(model_name, model, valloader, device):
    model.eval()
    with torch.no_grad():

        # Predicting
        for i, (images, masks, _) in enumerate(valloader):
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            h, w = masks.size(1), masks.size(2)
            ph, pw = logits[0].size(2), logits[0].size(3)
            if ph != h or pw != w:
                for j in range(len(logits)):
                    logits[j] = F.interpolate(logits[j], size=(h, w), mode='bilinear', align_corners=False)

            predictions = torch.argmax(torch.softmax(logits[-2], dim=1), dim=1)
            
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
            batch_size=args.batch_size,
            data_augmentation=True,
            args=args
        )

        trg_trainloader, trg_valloader, _ = dataset_preprocessing(
            domain="Rural",
            batch_size=args.batch_size,
            data_augmentation=True,
            args=args
        )
        
        # inspect_dataset(src_trainloader, src_valloader)

        model, model_D1, model_D2 = get_model(args, device)

        model_number = get_model_number(res_dir)
        if args.resume:
            model = load_model(model, f"{res_dir}/weights/last_{model_number-1}.pt", device)
            model_D2 = load_model(model_D2, f"{res_dir}/weights/last_D2_{model_number-1}.pt", device)
            if args.mode == "multi_level":
                model_D1 = load_model(model_D1, f"{res_dir}/weights/last_D1_{model_number-1}.pt", device)


        criterion, bd_criterion, bce_criterion = get_criterion()
        optimizer, optimizer_D1, optimizer_D2 = get_optimizer(model, model_D1, model_D2, args)
        scheduler, scheduler_D1, scheduler_D2 = get_scheduler(optimizer, optimizer_D1, optimizer_D2, args)

        if args.resume:
            for _ in range(args.resume_epoch-1):
                scheduler.step()
                scheduler_D2.step()
                if args.mode == "multi_level":
                    scheduler_D1.step()

        log_training_setup(device, args, train_monitor)

        if args.mode == "single_level":
            train_single_level(
                model=model,
                model_D2=model_D2,
                model_number=model_number,
                src_trainloader=src_trainloader,
                trg_trainloader=trg_trainloader,
                src_valloader=src_valloader,
                trg_valloader=trg_valloader,
                criterion=criterion,
                bd_criterion=bd_criterion,
                bce_criterion=bce_criterion,
                optimizer=optimizer,
                optimizer_D2=optimizer_D2,
                scheduler=scheduler,
                scheduler_D2=scheduler_D2,
                epochs=args.epochs,
                init_epoch=args.resume_epoch,
                patience=args.patience,
                device=device,
                monitor=train_monitor,
                res_dir=res_dir
            )
        elif args.mode == "multi_level":
            train_multi_level(
                model=model,
                model_D1=model_D1,
                model_D2=model_D2,
                model_number=model_number,
                src_trainloader=src_trainloader,
                trg_trainloader=trg_trainloader,
                src_valloader=src_valloader,
                trg_valloader=trg_valloader,
                criterion=criterion,
                bd_criterion=bd_criterion,
                bce_criterion=bce_criterion,
                optimizer=optimizer,
                optimizer_D1=optimizer_D1,
                optimizer_D2=optimizer_D2,
                scheduler=scheduler,
                scheduler_D1=scheduler_D1,
                scheduler_D2=scheduler_D2,
                epochs=args.epochs,
                init_epoch=args.resume_epoch,
                patience=args.patience,
                device=device,
                monitor=train_monitor,
                res_dir=res_dir
            )
    

    if args.test:
        res_dir = get_results_dir(args.store, args.model_name, args.version)

        file_name = f"{res_dir}/testing_log.txt"
        resume = os.path.exists(file_name)
        test_monitor = Monitor(file_name, resume, inline=False)

        trainloader, valloader, _ = dataset_preprocessing(
            domain=args.target_domain,
            batch_size=args.batch_size,
            data_augmentation=False,
            args=args
        )

        model, _, _ = get_model(args.model_name, device)
        model = load_model(model, f"{res_dir}/weights/{args.model_file}", device)

        log_testing_setup(device, args, test_monitor)

        test(
            model_name=args.model_name,
            model=model,
            valloader=valloader,
            device=device,
            monitor=test_monitor
        )

    
    if args.predict:
        res_dir = get_results_dir(args.store, args.model_name, args.version)

        trainloader, valloader, _ = dataset_preprocessing(
            domain=args.target_domain,
            batch_size=1,
            data_augmentation=False,
            args=args
        )

        model, _, _ = get_model(args.model_name, device)
        model = load_model(model, f"{res_dir}/weights/{args.model_file}", device)

        predict(
            model_name=args.model_name,
            model=model,
            valloader=valloader,
            device=device
        )



if __name__ == "__main__":
    main()
