import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.loaders import get_dataloaders


# Utility: Get model architecture
def get_student_model(student_name, num_classes):
    if student_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif student_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif student_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown student model: {student_name}")
    return model


# Evaluation function
def evaluate_student(model, val_loader):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return acc


if __name__ == "__main__":
    dataset = "dermamnist"
    num_classes = 7
    data_dir = "data"

    # Load val loader only
    _, val_loader, _ = get_dataloaders(data_dir, dataset, batch_size=64, image_size=28, num_workers=2)

    student_models = {
        "resnet18": f"models/resnet18/resnet18_dermamnist_student.pth",
        "mobilenet_v2": f"models/mobilenet_v2/mobilenet_v2_dermamnist_student.pth",
        "efficientnet_b0": f"models/efficientnet_b0/efficientnet_b0_dermamnist_student.pth"
    }

    print("📊 Student Model Evaluation on dermaMNIST\n")

    for name, path in student_models.items():
        if not os.path.exists(path):
            print(f"❌ {name}: Model file not found at {path}")
            continue

        print(f"🔍 Evaluating {name}...")

        model = get_student_model(name, num_classes).cuda()
        state_dict = torch.load(path, map_location="cuda")
        model.load_state_dict(state_dict)

        acc = evaluate_student(model, val_loader)
        print(f"✅ {name:<16} => Accuracy: {acc:.4f}\n")
