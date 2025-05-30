import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
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


def evaluate(dataset_name):
    info = INFO[dataset_name]
    num_classes = len(info['label'])
    is_multilabel = info['task'] == 'multi-label'

    # Load validation loader
    _, val_loader, _ = get_dataloaders('data', dataset_name, batch_size=64, image_size=28, num_workers=0)

    # Load model
    model_path = f"models/resnet50_teacher_{dataset_name}.pth"
    model = TeacherResNet50(num_classes).cuda()
    state_dict = torch.load(model_path, map_location='cuda', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Evaluating {dataset_name}"):
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)

            if is_multilabel:
                preds = torch.sigmoid(outputs)
            else:
                preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    if is_multilabel:
        # Compute per-class AUC to avoid sklearn mix-up
        aucs = []
        for i in range(num_classes):
            try:
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            except ValueError:
                pass
        metric = float(np.mean(aucs)) if aucs else 0.0
        return f"AUROC: {metric:.4f}"
    else:
        acc = accuracy_score(y_true, y_pred)
        return f"Accuracy: {acc:.4f}"


if __name__ == "__main__":
    datasets = ["pathmnist", "chestmnist", "organamnist", "dermamnist", "octmnist"]
    print("📊 Teacher Model Evaluation Results:\n")

    for ds in datasets:
        model_path = f"models/resnet50_teacher_{ds}.pth"
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found for {ds}")
            continue
        try:
            result = evaluate(ds)
            print(f"{ds.upper():<12} => {result}")
        except Exception as e:
            print(f"❌ Failed to evaluate {ds}: {e}")
