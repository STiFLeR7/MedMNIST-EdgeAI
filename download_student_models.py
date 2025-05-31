import os
import torch
from torchvision import models

# Target directory to save models
base_dir = "models"
os.makedirs(base_dir, exist_ok=True)

def save_model(model, name):
    model_dir = os.path.join(base_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.pth")
    torch.save(model.state_dict(), path)
    print(f"✅ Saved {name} to {path}")

# Download & save ResNet18
save_model(models.resnet18(weights=None), "resnet18")

# Download & save MobileNetV2
save_model(models.mobilenet_v2(weights=None), "mobilenet_v2")

# Download & save EfficientNet-B0
save_model(models.efficientnet_b0(weights=None), "efficientnet_b0")
