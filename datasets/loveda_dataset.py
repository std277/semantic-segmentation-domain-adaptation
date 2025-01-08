import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as F


class LoveDADataset(Dataset):
    def __init__(self, dataset_type, domain, transform, root_dir):
        """
        Args:
            dataset_type (str): Type of the dataset ('Train', 'Val', 'Test').
            domain (str): The domain of the dataset ('Urban', 'Rural').
            transform (callable): Transform to apply to the images.
            root_dir (str): Root directory of the dataset.
        """

        self.transform = transform
        self.images = []
        self.masks = []

        # Paths for urban training data
        image_dir = os.path.join(
            root_dir, f'{dataset_type}/{domain}/images_png')
        mask_dir = os.path.join(root_dir, f'{dataset_type}/{domain}/masks_png')

        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                mask_path = os.path.join(mask_dir, filename)

                self.images.append(image_path)
                self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # loading image and mask
        image = torchvision.io.read_image(
            self.images[idx]).float() / 255.0  # Normalize to [0, 1]
        mask = torchvision.io.read_image(
            self.masks[idx], mode=torchvision.io.ImageReadMode.GRAY)

        # resizing to 512x512 (consistent dimensions for model input)
        # image = torchvision.transforms.functional.resize(image, size=(512, 512))
        # mask = torchvision.transforms.functional.resize(mask, size=(512, 512), interpolation=Image.NEAREST)

        image = F.resize(image, size=(512, 512))
        mask = F.resize(mask, size=(512, 512), interpolation=Image.NEAREST)

        image = self.transform(image)

        return image, mask.long()
