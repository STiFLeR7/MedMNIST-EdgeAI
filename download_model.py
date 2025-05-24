import torch
import torchvision.models as models
import os

# Create local model directory
model_dir = os.path.join('models', 'resnet50')
os.makedirs(model_dir, exist_ok=True)

# Tell torch to cache here
torch.hub.set_dir(model_dir)

# Load pretrained model
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Save locally
save_path = os.path.join(model_dir, 'resnet50.pth')
torch.save(resnet50.state_dict(), save_path)

print(f"ResNet-50 saved at: {save_path}")
