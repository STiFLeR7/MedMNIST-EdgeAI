import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MedMNISTDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None):
        """
        Args:
            data_path (str): Path to the .npz file
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data = np.load(data_path)
        self.images = self.data[f'{split}_images']
        self.labels = self.data[f'{split}_labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to torch.Tensor and normalize to [0,1]
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        # Add channel dim if grayscale (for CNNs)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).squeeze().long()
