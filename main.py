import argparse
import os

import random

import numpy as np

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from torch.backends import cudnn
from torch.amp import GradScaler, autocast

from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2

from datasets import LoveDADataset
from models import DeepLabV2_ResNet101
from utils import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_results_dir(store, model_name, version):
    if store == "drive":
        res_dir = "/content/drive/MyDrive/res"
    else:
        res_dir = "res"

    os.makedirs(res_dir, exist_ok=True)

    dir_name = f"{model_name}_{version}"
    for file in os.listdir(res_dir):
        if file == dir_name:
            raise Exception(f"Directory {dir_name} already exists")

    res_dir = f"{res_dir}/{dir_name}"
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


def log_training_setup(model, loss_function, optimizer, scheduler, args, monitor):
    monitor.log(f"Model:\n{model}\n")

    monitor.log(f"Loss function:\n{loss_function}\n")

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


def dataset_preprocessing(domain, batch_size):
    # Define transforms
    transform = Compose([
        Resize(512, 512),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
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


def get_model(model_name, device):
    if model_name == "DeepLabV2_ResNet101":
        model = DeepLabV2_ResNet101(
            num_classes=7, pretrain=True, pretrain_model_path='./weights_pretrained/deeplab_resnet_pretrained_imagenet.pth')
    else:
        raise Exception(f"Model {model_name} doesn't exist")

    model = model.to(device)

    return model


def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name))
    return model


def get_loss_function():
    return nn.CrossEntropyLoss()


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











def train(model, trainloader, loss_function, optimizer, scheduler, epochs, device, monitor, res_dir):
    cudnn.benchmark = True

    if device.type == "cuda":
        scaler = GradScaler("cuda")
    else:
        scaler = None

    train_losses = []
    learning_rates = []

    for e in range(epochs):
        monitor.start(desc=f"Epoch {e + 1}/{epochs}", max_progress=len(trainloader))

        learning_rate = scheduler.get_last_lr()[0]
        learning_rates.append(learning_rate)

        train_loss = 0.0
        cumulative_loss = 0.0
        count_loss = 0

        model.train()
        for i, (images, masks) in enumerate(trainloader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            if scaler:
                with autocast("cuda"):
                    logits = model(images)
                    # outputs = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    loss = loss_function(logits, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                # outputs = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                loss = loss_function(logits, masks)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            cumulative_loss += loss.item()
            count_loss += 1
            train_loss = cumulative_loss / count_loss

            monitor.update(
                i + 1,
                learning_rate=f"{learning_rate:.5f}",
                train_loss=f"{train_loss:.4f}",
            )

        train_losses.append(train_loss)
        monitor.stop()

        scheduler.step()

        save_model(model, f"{res_dir}/weights/last.pt")

        plot_loss(train_losses, res_dir)
        plot_learning_rate(learning_rates, res_dir)

    monitor.print_stats()


def test(model, valloader, device, monitor):
    monitor.start(desc=f"Testing", max_progress=len(valloader))

    inference_times = []

    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(valloader):
            images, masks = images.to(device), masks.to(device)

            start_time = time.perf_counter()
            logits = model(images)
            end_time = time.perf_counter()

            batch_inference_time = (end_time - start_time) / images.size(0)
            inference_times.append(batch_inference_time)

            monitor.update(
                i + 1,
            )

    monitor.stop()

    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    # monitor.log(f"Accuracy on test images: {100 * test_accuracy:.3f} %")
    monitor.log(f"Mean inference time: {mean_inference_time * 1000:.3f} ms")
    monitor.log(f"Standard deviation of inference time: {std_inference_time * 1000:.3f} ms")











def main(args):
    set_seed(args.seed)
    device = get_device()

    model = get_model(args.model_name, device)

    if args.train:
        res_dir = make_results_dir(args.store, args.model_name, args.version)
        
        train_monitor = Monitor(file_name=f"{res_dir}/training_log.txt")

        trainloader, valloader, testloader = dataset_preprocessing(
            domain=args.source_domain,
            batch_size=args.batch_size
        )
        
        # inspect_dataset(trainloader, valloader, testloader)

        loss_function = get_loss_function()
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        log_training_setup(model, loss_function, optimizer, scheduler, args, train_monitor)

        train(
            model=model,
            trainloader=trainloader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            device=device,
            monitor=train_monitor,
            res_dir=res_dir
        )




    if args.test:
        res_dir = get_results_dir(args.store, args.model_name, args.version)

        test_monitor = Monitor(file_name=f"{res_dir}/testing_log.txt")


        test_monitor.log(f"Model: {args.model_name}")

        test(
            model=model,
            valloader=valloader,
            device=device,
            monitor=test_monitor
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR10 Classification")

    domains_choices = [
        "Rural",
        "Urban"
    ]

    models_choices = [
        "DeepLabV2_ResNet101",
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
        help="Enable training mode"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable testing mode"
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
        type=int,
        default=0,
        help=f"Specify the version.",
    )

    parser.add_argument(
        "--test_model_file",
        type=str,
        default="best.pt",
        help=f"Specify the model file name for testing.",
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
        "--batch_size",
        type=int,
        default=8,
        help=f"Specify the batch size.",
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
    main(args)
