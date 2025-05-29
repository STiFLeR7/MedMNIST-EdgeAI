import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MedMNISTDataset(Dataset):
    def __init__(self, path, split='train', transform=None):
        data = np.load(path)
        self.images = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"]
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        label = int(label)  # <- 🔥 FIX HERE

        image = image.astype(np.uint8)
        if image.ndim == 2:
            image = Image.fromarray(image, mode='L')
        else:
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)
