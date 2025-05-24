import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.loaders import get_dataloaders
from utils.config import load_config
import numpy as np

class TeacherResNet50(nn.Module):
    def __init__(self, num_classes, weights_path=None):
        super(TeacherResNet50, self).__init__()
        self.model = models.resnet50(weights=None)  # prevent downloading
        if weights_path and os.path.isfile(weights_path):
            print(f"🔍 Loading local weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"❌ Local ResNet50 weights not found at: {weights_path}")

        # Replace final FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_teacher():
    # Load config
    cfg = load_config()
    dataset = cfg['data']['dataset']
    data_dir = cfg['data']['dir']
    batch_size = cfg['data']['batch_size']
    image_size = cfg['data']['image_size']
    num_workers = cfg['data']['num_workers']

    lr = cfg['train']['lr']
    epochs = cfg['train']['epochs']
    weight_decay = cfg['train']['weight_decay']
    optimizer_choice = cfg['train']['optimizer']
    save_dir = cfg['log']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Path to local ResNet-50 pretrained weights
    weights_path = os.path.join("models", "resnet50", "resnet50.pth")

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(data_dir, dataset, batch_size, image_size, num_workers)
    labels = [label for _, label in train_loader.dataset]
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Init model with local weights
    model = TeacherResNet50(num_classes, weights_path).cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_choice.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer in config")

    best_acc = 0
    best_model_path = os.path.join(save_dir, f"resnet50_teacher_{dataset}.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Training"):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at {best_model_path} with Val Acc: {val_acc:.4f}")

    print(f"🏁 Training complete. Highest Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train_teacher()
