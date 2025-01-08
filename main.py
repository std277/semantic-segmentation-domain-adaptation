import argparse
import os

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.utils import make_grid

from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast, Resize, Normalize, Compose
)
from albumentations.pytorch import ToTensorV2

from datasets import LoveDADataset


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


def overlap_mask_on_image(image, mask):
    pass

def plot_image(image, mask=None, title=None, show=True):
    plt.figure()

    if title is not None:
        plt.title(title)

    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))

    if mask is not None:
        np_mask = mask.numpy()
        
        plt.imshow(np_mask, cmap='jet', alpha=0.2)

    plt.axis("off")

    if show:
        plt.show()


def plot_batch():
    pass


def inspect_dataset(trainloader, valloader, testloader):
    def imshow(title, img):
        # img = img / 2 + 0.5
        npimg = img.numpy()
        plt.figure()
        plt.title(title)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")


    for type, loader in zip(("Train", "Val", "Test"), (trainloader, valloader, testloader)):
        it = iter(loader)
        images, masks = next(it)

        # Overlap faded mask on image

        imshow(title=f"{type} sample batch", img=make_grid(images))

    plt.show()



def main(args):
    torch.manual_seed(args.seed)

    device = get_device()

    if  args.store == "drive":
        res_dir = "/content/drive/MyDrive/res"
    else:
        res_dir = "res"

    os.makedirs(res_dir, exist_ok=True)

    trainloader, valloader, testloader = dataset_preprocessing(domain="Urban", batch_size=4)

    # inspect_dataset(trainloader, valloader, testloader)

    trainit = iter(trainloader)
    images, masks = next(trainit)

    for image, mask in zip(images, masks):
        plot_image(image, mask, title="Urban train image", show=False)
    plt.show()





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
