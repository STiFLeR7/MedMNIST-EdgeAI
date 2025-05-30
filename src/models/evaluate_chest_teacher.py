import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from medmnist import INFO
from src.loaders import get_dataloaders
import numpy as np

class TeacherResNet50(nn.Module):
    def __init__(self, num_classes):
        super(TeacherResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def evaluate_chest():
    dataset = 'chestmnist'
    info = INFO[dataset]
    num_classes = len(info['label'])

    # Load validation set
    _, val_loader, _ = get_dataloaders('data', dataset, batch_size=64, image_size=28, num_workers=0)

    # Load model
    model_path = f"models/resnet50_teacher_{dataset}.pth"
    model = TeacherResNet50(num_classes).cuda()
    state_dict = torch.load(model_path, map_location='cuda', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    all_probs, all_targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating ChestMNIST"):
            images = images.cuda()
            outputs = torch.sigmoid(model(images)).cpu().numpy()
            labels = labels.numpy().astype('float32')

            all_probs.append(outputs)
            all_targets.append(labels)

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # AUROC: mean over all labels
    aucs = []
    for i in range(num_classes):
        if len(np.unique(y_true[:, i])) > 1:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            aucs.append(auc)
    auroc = np.mean(aucs)

    print(f"\n✅ ChestMNIST AUROC: {auroc:.4f}")

if __name__ == "__main__":
    print("📊 Evaluating Teacher on ChestMNIST (multi-label AUROC)...")
    evaluate_chest()
