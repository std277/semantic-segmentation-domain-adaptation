import argparse
import os

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.utils import make_grid

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
    dataset_transform = T.Compose([
        T.Resize(512),
        T.ToTensor()
    ])

    # Define the Dataset object for training, validation and testing
    traindataset = LoveDADataset(dataset_type="Train", domain=domain, transform=dataset_transform, root_dir='data')
    valdataset = LoveDADataset(dataset_type="Val", domain=domain, transform=dataset_transform, root_dir='data')
    testdataset = LoveDADataset(dataset_type="Test", domain=domain, transform=dataset_transform, root_dir='data')

    # Define the DataLoaders
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, num_workers=2)
    testloader = DataLoader(testdataset, batch_size=batch_size, num_workers=2)

    return trainloader, valloader, testloader


def overlap_mask_on_image(image, mask):
    pass

def plot_image():
    pass

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


    for type, loader in zip(
        ("Train", "Val", "Test"), (trainloader, valloader, testloader)
    ):
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
