import argparse
import os
import re

import random

import numpy as np

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from torch.backends import cudnn
from torch.amp import GradScaler, autocast

from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast, CoarseDropout
from albumentations.pytorch import ToTensorV2

from fvcore.nn import FlopCountAnalysis, flop_count_table

from datasets import LoveDADataset, MEAN, STD, NUM_CLASSES
from models import DeepLabV2_ResNet101, PIDNet_S, PIDNet_M, PIDNet_L
from criteria import CrossEntropyLoss, OhemCrossEntropyLoss
from utils import *




def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR10 Classification")

    domains_choices = [
        "Rural",
        "Urban"
    ]

    models_choices = [
        "DeepLabV2_ResNet101",
        "PIDNet_S",
        "PIDNet_M",
        "PIDNet_L",
    ]

    criteria_choices = [
        "CrossEntropyLoss",
        "OhemCrossEntropyLoss"
    ]

    optimizers_choices = [
        "Adam",
        "SGD"
    ]

    schedulers_choices = [
        "ConstantLR",
        "StepLR",
        "CosineAnnealingLR",
        "PolynomialLR"
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
        "--source_domain",
        type=str,
        choices=domains_choices,
        default="Rural",
        help=f"Specify the source domain for training.",
    )

    parser.add_argument(
        "--target_domain",
        type=str,
        choices=domains_choices,
        default="Rural",
        help=f"Specify the target domain for testing.",
    )

    parser.add_argument(
        "--data_augmentation",
        action="store_true",
        help="Performs data augmentation on dataset."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=f"Specify the batch size.",
    )

    parser.add_argument(
        "--criterion",
        type=str,
        choices=criteria_choices,
        default="CrossEntropyLoss",
        help=f"Specify the criterion.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=optimizers_choices,
        default="SGD",
        help=f"Specify the optimizer.",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        choices=schedulers_choices,
        default="CosineAnnealingLR",
        help=f"Specify the learning rate scheduler.",
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
        "--step_size",
        type=int,
        default=10,
        help=f"Specify the step size for StepLR scheduler.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help=f"Specify gamma for StepLR scheduler.",
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

    dir_name = f"{model_name}_{version}"
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

    dir_name = f"{model_name}_{version}"
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


def log_training_setup(model, criterion, optimizer, scheduler, device, args, monitor):
    monitor.log(f"Model: {args.model_name}")

    monitor.log(f"Device: {device}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        monitor.log(f"Cuda device name: {device_name}")


    monitor.log(f"Dataset source domain: {args.source_domain}")

    monitor.log(f"Data augmentation: {args.data_augmentation}")

    monitor.log(f"Batch size: {args.batch_size}\n")

    monitor.log(f"Criterion: {args.criterion}\n")

    monitor.log(f"Optimizer:\n{args.optimizer} (")
    if args.optimizer == "Adam":
        monitor.log(f"    lr: {args.lr}")
    elif args.optimizer == "SGD":
        monitor.log(f"    lr: {args.lr}")
        monitor.log(f"    momentum: {args.momentum}")
        monitor.log(f"    weight_decay: {args.weight_decay}")
    monitor.log(")\n")

    monitor.log(f"Scheduler:\n{args.scheduler} (")
    if args.scheduler == "ConstantLR":
        monitor.log(f"    lr: {args.lr}")
    elif args.scheduler == "StepLR":
        monitor.log(f"    lr: {args.lr}")
        monitor.log(f"    step_size: {args.step_size}")
        monitor.log(f"    gamma: {args.gamma}")
    elif args.scheduler == "CosineAnnealingLR":
        monitor.log(f"    lr: {args.lr}")
        monitor.log(f"    t_max: {args.epochs}")
    elif args.scheduler == "PolynomialLR":
        monitor.log(f"    lr: {args.lr}")
        monitor.log(f"    power: {args.power}")
    monitor.log(")\n")



def log_testing_setup(device, args, monitor):
    monitor.log(f"Model test file: {args.model_file}")
    monitor.log(f"Device: {device}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        monitor.log(f"Cuda device name: {device_name}")
    monitor.log(f"Dataset target domain: {args.target_domain}\n")


def dataset_preprocessing(domain, batch_size, data_augmentation, model_name):
    # Define transforms
    augmentation_transform = Compose([
        Resize(512, 512),
        Normalize(mean=MEAN, std=STD),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.3),
        CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        ToTensorV2()
    ])

    transform = Compose([
        Resize(512, 512),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    # Define the Dataset object for training, validation and testing
    traindataset = LoveDADataset(dataset_type="Train", domain=domain, transform=(augmentation_transform if data_augmentation else transform), root_dir='data')
    valdataset = LoveDADataset(dataset_type="Val", domain=domain, transform=transform, root_dir='data')
    testdataset = LoveDADataset(dataset_type="Test", domain=domain, transform=transform, root_dir='data')

    # Define the DataLoaders
    trainloader = DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, num_workers=2)
    testloader = DataLoader(testdataset, batch_size=batch_size, num_workers=2)

    return trainloader, valloader, testloader


def get_model(model_name, device):
    if model_name == "DeepLabV2_ResNet101":
        model = DeepLabV2_ResNet101(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path='./weights_pretrained/deeplab_resnet_pretrained_imagenet.pth')
    elif model_name == "PIDNet_S":
        model = PIDNet_S(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_s_pretrained_imagenet.pth")
    elif model_name == "PIDNet_M":
        model = PIDNet_M(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_m_pretrained_imagenet.pth")
    elif model_name == "PIDNet_L":
        model = PIDNet_L(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path="./weights_pretrained/pidnet_l_pretrained_imagenet.pth")
    else:
        raise Exception(f"Model {model_name} doesn't exist")

    model = model.to(device)

    return model


def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


def load_model(model, file_name, device):
    model.load_state_dict(torch.load(file_name, map_location=torch.device(device), weights_only=True))
    return model


def get_criterion(args):
    if args.criterion == "CrossEntropyLoss":
        criterion = CrossEntropyLoss(ignore_label=255)
    elif args.criterion == "OhemCrossEntropyLoss":
        criterion = OhemCrossEntropyLoss(ignore_label=255, thres=0.9, min_kept=131072)
    else:
        raise Exception(f"Criterion {args.criterion} doesn't exist")
    return criterion



def get_optimizer(model, args):
    if args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise Exception(f"Optimizer {args.optimizer} doesn't exist")
    
    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler == "ConstantLR":
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0
        )
    elif args.scheduler == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    elif args.scheduler == "PolynomialLR":
        max_iters = args.epochs
        power = args.power

        def polynomial_lr(current_iter):
            return (1 - current_iter / max_iters)**power
        
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=polynomial_lr
        )
    else:
        raise Exception(f"Scheduler {args.scheduler} doesn't exist")

    return scheduler


def compute_mIoU(predictions, masks, num_classes):
    iou_per_class = torch.zeros(num_classes, dtype=torch.float32)

    predictions = predictions.view(-1)
    masks = masks.view(-1)

    for cls in range(num_classes):
        intersection = torch.sum((predictions == cls) & (masks == cls))
        union = torch.sum((predictions == cls) | (masks == cls))

        if union == 0:
            iou_per_class[cls] = float('nan')
        else:
            iou_per_class[cls] = intersection / union

    mean_iou = torch.nanmean(iou_per_class).item()

    return mean_iou







def train(model_name, model, model_number, trainloader, valloader, criterion, optimizer, scheduler, epochs, init_epoch, patience, device, monitor, res_dir):
    cudnn.benchmark = True

    train_losses = []
    val_losses = []
    train_mIoUs = []
    val_mIoUs = []
    learning_rates = []

    best_val_loss = None
    patience_counter = 0

    for e in range(init_epoch-1, epochs):
        # Training
        monitor.start(desc=f"Epoch {e + 1}/{epochs}", max_progress=len(trainloader))

        learning_rate = scheduler.get_last_lr()[0]
        learning_rates.append(learning_rate)

        cumulative_loss = 0.0
        cumulative_mIoU = 0.0
        count = 0
        train_mIoU = 0.0

        model.train()
        for i, (images, masks) in enumerate(trainloader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            if model_name in ("DeepLabV2_ResNet101",):
                logits = model(images)
                loss = criterion([logits], masks)

            elif model_name in ("PIDNet_S", "PIDNet_M", "PIDNet_L"):
                logits = model(images)

                h, w = masks.size(1), masks.size(2)
                ph, pw = logits[0].size(2), logits[0].size(3)
                if ph != h or pw != w:
                    for j in range(len(logits)):
                        logits[j] = F.interpolate(logits[j], size=(h, w), mode='bilinear', align_corners=False)

                loss_s = criterion(logits[:-1], masks, balance_weights=[0.4, 1.0])

                filler = torch.ones_like(masks) * 255
                bd_label = torch.where(F.sigmoid(logits[-1][:,0,:,:])>0.8, masks, filler)
                loss_sb = criterion([logits[-2]], bd_label)
                
                loss = loss_s + loss_sb

                logits = logits[-2]

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            count += 1

            cumulative_loss += loss.item()
            train_loss = cumulative_loss / count

            mIoU = compute_mIoU(predictions, masks, NUM_CLASSES)
            cumulative_mIoU += mIoU
            train_mIoU = cumulative_mIoU / count

            monitor.update(
                i + 1,
                learning_rate=f"{learning_rate:.5f}",
                train_loss=f"{train_loss:.4f}",
                train_mIoU=f"{train_mIoU:.4f}",
            )

        train_losses.append(train_loss)
        train_mIoUs.append(train_mIoU)

        monitor.stop()




        # Validation
        monitor.start(desc=f"Validation", max_progress=len(valloader))

        cumulative_loss = 0.0
        cumulative_mIoU = 0.0
        count = 0
        val_mIoU = 0.0

        model.eval()
        with torch.no_grad():
            for i, (images, masks) in enumerate(valloader):
                images, masks = images.to(device), masks.to(device)
                if model_name in ("DeepLabV2_ResNet101",):
                    logits = model(images)
                    loss = criterion([logits], masks)

                elif model_name in ("PIDNet_S", "PIDNet_M", "PIDNet_L"):
                    logits = model(images)

                    h, w = masks.size(1), masks.size(2)
                    ph, pw = logits[0].size(2), logits[0].size(3)
                    if ph != h or pw != w:
                        for j in range(len(logits)):
                            logits[j] = F.interpolate(logits[j], size=(h, w), mode='bilinear', align_corners=False)

                    loss_s = criterion(logits[:-1], masks, balance_weights=[0.4, 1.0])

                    filler = torch.ones_like(masks) * 255
                    bd_label = torch.where(F.sigmoid(logits[-1][:,0,:,:])>0.8, masks, filler)
                    loss_sb = criterion([logits[-2]], bd_label)
                
                    loss = loss_s + loss_sb

                    logits = logits[-2]
                
                predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                
                count += 1
                
                cumulative_loss += loss.item()
                val_loss = cumulative_loss / count

                mIoU = compute_mIoU(predictions, masks, NUM_CLASSES)
                cumulative_mIoU += mIoU
                val_mIoU = cumulative_mIoU / count
            
                monitor.update(
                    i + 1,
                    val_loss=f"{val_loss:.4f}",
                    val_mIoU=f"{val_mIoU:.4f}",
                )

        val_losses.append(val_loss)
        val_mIoUs.append(val_mIoU)

        monitor.stop()

        if best_val_loss is None or val_loss < best_val_loss:
            save_model(model, f"{res_dir}/weights/best_{model_number}.pt")
            monitor.log(f"Model saved as best_{model_number}.pt\n")
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            monitor.log(f"Early stopping after {e + 1} epochs\n")
            break


        scheduler.step()

        save_model(model, f"{res_dir}/weights/last_{model_number}.pt")

        plot_loss(train_losses, val_losses, model_number, res_dir)
        plot_mIoU(train_mIoUs, val_mIoUs, model_number, res_dir)
        plot_learning_rate(learning_rates, model_number, res_dir)


    monitor.print_stats()


def test(model_name, model, valloader, device, monitor):
    monitor.start(desc=f"Testing", max_progress=len(valloader))

    flops_count = 0
    cumulative_mIoU = 0.0
    count = 0
    test_mIoU = 0.0
    inference_times = []

    model.eval()
    with torch.no_grad():
        
        # FLOPs analysis
        images, _ = next(iter(valloader))
        images = images.to(device)
        flops = FlopCountAnalysis(model, images)
        flops_count = flop_count_table(flops)

        # Testing
        for i, (images, masks) in enumerate(valloader):
            images, masks = images.to(device), masks.to(device)

            if model_name in ("DeepLabV2_ResNet101",):
                start_time = time.perf_counter()
                logits = model(images)
                end_time = time.perf_counter()
            elif model_name in ("PIDNet_S", "PIDNet_M", "PIDNet_L"):
                start_time = time.perf_counter()
                logits = model(images)
                end_time = time.perf_counter()

                h, w = masks.size(1), masks.size(2)
                ph, pw = logits[0].size(2), logits[0].size(3)
                if ph != h or pw != w:
                    for j in range(len(logits)):
                        logits[j] = F.interpolate(logits[j], size=(h, w), mode='bilinear', align_corners=False)

                logits = logits[-2]

            batch_inference_time = (end_time - start_time) / images.size(0)
            inference_times.append(batch_inference_time)

            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            count += 1

            mIoU = compute_mIoU(predictions, masks, NUM_CLASSES)
            cumulative_mIoU += mIoU
            test_mIoU = cumulative_mIoU / count
            
            monitor.update(
                i + 1,
                test_mIoU=f"{test_mIoU:.4f}",
            )

    monitor.stop()

    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    monitor.log(f"FLOPs:\n{flops_count}\n")
    monitor.log(f"Mean Intersection over Union on test images: {test_mIoU*100:.3f} %")
    monitor.log(f"Mean inference time: {mean_inference_time * 1000:.3f} ms")
    monitor.log(f"Standard deviation of inference time: {std_inference_time * 1000:.3f} ms")





def predict(model_name, model, valloader, device):
    model.eval()
    with torch.no_grad():

        # Predicting
        for i, (images, masks) in enumerate(valloader):
            images, masks = images.to(device), masks.to(device)

            if model_name in ("DeepLabV2_ResNet101",):
                logits = model(images)
            elif model_name in ("PIDNet_S", "PIDNet_M", "PIDNet_L"):
                logits = model(images)
                h, w = masks.size(1), masks.size(2)
                ph, pw = logits[0].size(2), logits[0].size(3)
                if ph != h or pw != w:
                    for j in range(len(logits)):
                        logits[j] = F.interpolate(logits[j], size=(h, w), mode='bilinear', align_corners=False)

                logits = logits[-2]

            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            mIoU = compute_mIoU(predictions, masks, NUM_CLASSES)

            plot_prediction(images[0], masks[0], predictions[0], alpha=0.4, title="Prediction", description=f"mIoU: {mIoU*100:.3f} %")

            












def main():
    args = parse_args()

    if args.train + args.test + args.predict > 1:
        raise Exception("Both train and test arguments are selected")

    set_seed(args.seed)
    device = get_device()

    if args.train:
        res_dir = make_results_dir(args.store, args.model_name, args.version, args.resume)

        file_name = f"{res_dir}/training_log.txt"
        train_monitor = Monitor(file_name, resume=args.resume)

        trainloader, valloader, _ = dataset_preprocessing(
            domain=args.source_domain,
            batch_size=args.batch_size,
            data_augmentation=args.data_augmentation,
            model_name=args.model_name
        )
        
        # inspect_dataset(trainloader, valloader, testloader)
        # inspect_dataset_masks(trainloader, valloader, testloader)

        model = get_model(args.model_name, device)

        model_number = get_model_number(res_dir)
        if args.resume:
            model = load_model(model, f"{res_dir}/weights/last_{model_number-1}.pt", device)

        criterion = get_criterion(args)
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        if args.resume:
            for _ in range(args.resume_epoch-1):
                scheduler.step()

        log_training_setup(model, criterion, optimizer, scheduler, device, args, train_monitor)

        train(
            model_name=args.model_name,
            model=model,
            model_number=model_number,
            trainloader=trainloader,
            valloader=valloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
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
        test_monitor = Monitor(file_name, resume)

        trainloader, valloader, _ = dataset_preprocessing(
            domain=args.target_domain,
            batch_size=args.batch_size,
            data_augmentation=False,
            model_name=args.model_name
        )

        model = get_model(args.model_name, device)
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
            model_name=args.model_name
        )

        model = get_model(args.model_name, device)
        model = load_model(model, f"{res_dir}/weights/{args.model_file}", device)

        predict(
            model_name=args.model_name,
            model=model,
            valloader=valloader,
            device=device
        )



if __name__ == "__main__":
    main()
