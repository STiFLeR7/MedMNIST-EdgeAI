# src/models/teacher_template.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.loaders import get_dataloaders
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

class TeacherResNet50(nn.Module):
    def __init__(self, num_classes):
        super(TeacherResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_teacher(dataset, num_classes, is_multilabel):
    print(f"\n🚀 Training Teacher on {dataset.upper()}...")
    data_dir = 'data'
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_dataloaders(data_dir, dataset, batch_size=64, image_size=28, num_workers=0)

    # Model
    model = TeacherResNet50(num_classes).cuda()

    # Loss
    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_metric = 0
    best_path = os.path.join(save_dir, f"resnet50_teacher_{dataset}.pth")

    for epoch in range(20):
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/20]"):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)

            loss = criterion(outputs, labels.float() if is_multilabel else labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Evaluation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                all_preds.append(outputs.cpu())
                all_targets.append(labels.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        if is_multilabel:
            probas = torch.sigmoid(preds).numpy()
            targets = targets.numpy()
            metric = roc_auc_score(targets, probas, average='macro')
            print(f"📊 Val AUROC: {metric:.4f} | Train Loss: {avg_loss:.4f}")
        else:
            acc = (preds.argmax(dim=1) == targets).float().mean().item()
            metric = acc
            print(f"📊 Val Acc: {metric:.4f} | Train Loss: {avg_loss:.4f}")

        # Save best model
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best model to {best_path}")

    print(f"🏁 Finished training {dataset.upper()} | Best Metric: {best_metric:.4f}")
