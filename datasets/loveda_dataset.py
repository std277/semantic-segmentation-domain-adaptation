import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms import functional as F


class LoveDADataset(Dataset):
    def __init__(self, dataset_type, domain, transform, root_dir):
        """
        Args:
            dataset_type (str): Type of the dataset ('Train', 'Val', 'Test').
            domain (str): The domain of the dataset ('Urban', 'Rural').
            transform (callable): Transform to apply to the images.
            root_dir (str): Root directory of the dataset.
        """
        assert dataset_type in ['Train', 'Val', 'Test'], "Invalid dataset type"
        assert domain in ['Urban', 'Rural'], "Invalid domain"

        self.dataset_type = dataset_type
        self.transform = transform
        self.samples = []

        image_dir = os.path.join(root_dir, f'{dataset_type}/{domain}/images_png')
        mask_dir = os.path.join(root_dir, f'{dataset_type}/{domain}/masks_png')

        for filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            self.samples.append((image_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]

        image = Image.open(img_path).convert('RGB')
        mask = torch.zeros((512, 512))
        if self.dataset_type != 'Test':
            mask = Image.open(mask_path).convert('L')


        # loading image and mask
        # image = torchvision.io.read_image(
        #     self.images[index]).float() / 255.0  # Normalize to [0, 1]
        # mask = torchvision.io.read_image(
        #     self.masks[index], mode=torchvision.io.ImageReadMode.GRAY)

        # resizing to 512x512 (consistent dimensions for model input)
        # image = torchvision.transforms.functional.resize(image, size=(512, 512))
        # mask = torchvision.transforms.functional.resize(mask, size=(512, 512), interpolation=Image.NEAREST)

        # image = F.resize(image, size=(512, 512))
        # mask = F.resize(mask, size=(512, 512), interpolation=Image.NEAREST)

        image = self.transform(image)
        if self.dataset_type != 'Test':
            mask = self.transform(mask)

        return image, mask
