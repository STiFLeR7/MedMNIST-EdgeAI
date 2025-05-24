import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
from tqdm import tqdm

from src.loaders import get_dataloaders
from utils.config import load_config


class TeacherResNet50(nn.Module):
    def __init__(self, num_classes):
        super(TeacherResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(data_dir, dataset, batch_size, image_size, num_workers)
    num_classes = len(train_loader.dataset.class_names)

    # Init model
    model = TeacherResNet50(num_classes).cuda()

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
