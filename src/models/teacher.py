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
        self.model = models.resnet50(weights=None)
        if weights_path and os.path.isfile(weights_path):
            print(f"🔍 Loading local weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"❌ Local ResNet50 weights not found at: {weights_path}")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_teacher():
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

    weights_path = os.path.join("models", "resnet50", "resnet50.pth")

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(data_dir, dataset, batch_size, image_size, num_workers)

    # Get number of classes from the label shape
    sample_label = next(iter(train_loader))[1][0]
    is_multilabel = sample_label.ndim > 0 and sample_label.numel() > 1
    num_classes = sample_label.numel() if is_multilabel else len(np.unique([label for _, label in train_loader.dataset]))

    # Init model
    model = TeacherResNet50(num_classes, weights_path).cuda()

    # Criterion
    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()

    # Optimizer
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
            loss = criterion(outputs, labels.float() if is_multilabel else labels)

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

                if is_multilabel:
                    preds = torch.sigmoid(outputs) > 0.5
                    correct += (preds == labels.bool()).sum().item()
                    total += torch.numel(labels)
                else:
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at {best_model_path} with Val Acc: {val_acc:.4f}")

    print(f"🏁 Training complete. Highest Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train_teacher()
