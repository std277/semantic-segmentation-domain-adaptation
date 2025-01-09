import argparse
import os

import torch
from torch.utils.data import DataLoader

from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast, Resize, Normalize, Compose
)
from albumentations.pytorch import ToTensorV2

from datasets import LoveDADataset
from utils.plot import *


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


def dataset_preprocessing(domain, batch_size):
    # Define transforms
    transform = Compose([
        Resize(512, 512),
        # Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    # Define the Dataset object for training, validation and testing
    traindataset = LoveDADataset(dataset_type="Train", domain=domain, transform=transform, root_dir='data')
    valdataset = LoveDADataset(dataset_type="Val", domain=domain, transform=transform, root_dir='data')
    testdataset = LoveDADataset(dataset_type="Test", domain=domain, transform=transform, root_dir='data')

    # Define the DataLoaders
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, num_workers=2)
    testloader = DataLoader(testdataset, batch_size=batch_size, num_workers=2)

    return trainloader, valloader, testloader



def main(args):
    torch.manual_seed(args.seed)

    device = get_device()

    if  args.store == "drive":
        res_dir = "/content/drive/MyDrive/res"
    else:
        res_dir = "res"

    os.makedirs(res_dir, exist_ok=True)

    trainloader, valloader, testloader = dataset_preprocessing(domain="Urban", batch_size=4)

    inspect_dataset(trainloader, valloader, testloader)

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR10 Classification")

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help=f"Specify the random seed.",
    )

    parser.add_argument(
        "--store",
        type=str,
        choices=["local", "drive"],
        default="drive",
        help=f"Specify where to store results (local, drive).",
    )

    args = parser.parse_args()
    main(args)
