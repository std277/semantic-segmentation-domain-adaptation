import os
import cv2

from enum import Enum

import numpy as np

import torch
from torch.utils.data import Dataset

NUM_CLASSES = 7
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class LoveDADatasetLabel(Enum):
    BACKGROUND = 0
    BUILDING = 1
    ROAD = 2
    WATER = 3
    BARREN = 4
    FOREST = 5
    AGRICULTURE = 6


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

        image_dir = os.path.join(
            root_dir, f'{dataset_type}/{domain}/images_png')
        mask_dir = os.path.join(root_dir, f'{dataset_type}/{domain}/masks_png')

        for filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            self.samples.append((image_path, mask_path))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.dataset_type != 'Test':
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = mask - 1
        else:
            mask = torch.ones((512, 512), dtype=torch.long) * 255

        if self.dataset_type != 'Test':
            transformation = self.transform(image=image, mask=mask)
            image, mask = transformation['image'], transformation['mask']
            mask = mask.long()
        else:
            transformation = self.transform(image=image)
            image = transformation['image']

        edge = cv2.Canny(mask.numpy().astype(np.uint8), 0.1, 0.2)
        edge = torch.from_numpy(edge.astype(float))

        return image, mask, edge



def compute_boundaries(masks):
    edges = []
    for mask in masks:
        edge = cv2.Canny(mask.numpy().astype(np.uint8), 0.1, 0.2)
        edge = torch.from_numpy(edge.astype(float))
        edges.append(edge)

    edges = torch.stack(edges)
    return edges