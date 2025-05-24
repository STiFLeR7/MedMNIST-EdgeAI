import os
from medmnist.dataset import (
    PathMNIST,
    ChestMNIST,
    OrganMNISTAxial,
    DermaMNIST,
    OCTMNIST
)

# Custom root directory for dataset downloads
root_dir = os.path.join(os.getcwd(), 'data')

# Dataset mapping with correct classes
datasets = {
    'pathmnist': PathMNIST,
    'chestmnist': ChestMNIST,
    'organmnist_axial': OrganMNISTAxial,  # use the Axial version
    'dermamnist': DermaMNIST,
    'octmnist': OCTMNIST,
}

# Download train/val/test splits for each
for name, DatasetClass in datasets.items():
    print(f"\n📦 Downloading {name} into {root_dir}...")
    for split in ['train', 'val', 'test']:
        DatasetClass(split=split, root=root_dir, download=True)

print("\n✅ All selected datasets have been downloaded into:", root_dir)
