import os
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataloader import MedMNISTDataset
import numpy as np

def get_dataloaders(data_dir, dataset_name, batch_size=64, image_size=28, num_workers=2):
    data_path = os.path.join(data_dir, f"{dataset_name}.npz")

    # Auto-detect channel type (RGB or Grayscale) from sample image
    npz_data = np.load(data_path)
    sample_image = npz_data['train_images'][0]
    is_rgb = len(sample_image.shape) == 3 and sample_image.shape[2] == 3

    # Define transform
    if is_rgb:
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB
    else:
        normalize = transforms.Normalize([0.5], [0.5])  # Grayscale

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        normalize
    ])

    # Datasets
    train_dataset = MedMNISTDataset(data_path, split='train', transform=transform)
    val_dataset   = MedMNISTDataset(data_path, split='val', transform=transform)
    test_dataset  = MedMNISTDataset(data_path, split='test', transform=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
