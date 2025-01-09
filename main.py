import argparse
import os

import random

import torch
from torch.utils.data import DataLoader

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
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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



def get_model(model_name, device):
    if model_name == "DeepLabV2_ResNet101":
        model = DeepLabV2_ResNet101(num_classes=7, pretrain=True, pretrain_model_path='./weights_pretrained/deeplab_resnet_pretrained_imagenet.pth')
    else:
        raise Exception(f"Model {model_name} doesn't exist")

    model = model.to(device)

    return model


def make_results_dir(args):
    if  args.store == "drive":
        res_dir = "/content/drive/MyDrive/res"
    else:
        res_dir = "res"

    os.makedirs(res_dir, exist_ok=True)

    dir_name = f"{args.model_name}_{args.version}"
    for file in os.listdir(f"res"):
        if file == dir_name:
            raise Exception(f"Directory {dir_name} already exists")

    res_dir = f"{res_dir}/{dir_name}"
    sub_dirs = [res_dir, f"{res_dir}/weights", f"{res_dir}/plots"]
    for sub_dir in sub_dirs:
        os.makedirs(sub_dir, exist_ok=True)


def main(args):
    set_seed(args.seed)
    make_results_dir(args)
    device = get_device()

    trainloader, valloader, testloader = dataset_preprocessing(domain="Urban", batch_size=4)
    # inspect_dataset(trainloader, valloader, testloader)

    model = get_model(args.model_name, device)

    if args.train:
        pass

    if args.test:
        pass
    

    

    

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR10 Classification")

    models_choices = [
        "DeepLabV2_ResNet101",
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
