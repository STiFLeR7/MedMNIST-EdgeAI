import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
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


# Evaluation function for multi-label
def evaluate_student(model, val_loader):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.cuda()
            labels = labels.float().cuda()

            outputs = model(images)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    try:
        auroc = roc_auc_score(y_true, y_pred, average="macro")
        return auroc
    except Exception as e:
        print("❌ AUROC computation failed:", e)
        return 0.0


if __name__ == "__main__":
    dataset = "chestmnist"
    num_classes = 14
    data_dir = "data"

    # Lower batch size and workers for stability
    _, val_loader, _ = get_dataloaders(data_dir, dataset, batch_size=32, image_size=28, num_workers=0)

    student_models = {
        "resnet18": f"models/resnet18/resnet18_chestmnist_student.pth",
        "mobilenet_v2": f"models/mobilenet_v2/mobilenet_v2_chestmnist_student.pth",
        "efficientnet_b0": f"models/efficientnet_b0/efficientnet_b0_chestmnist_student.pth"
    }

    print("📊 Student Model Evaluation on CHESTMNIST (Multi-label AUROC)\n")

    for name, path in student_models.items():
        if not os.path.exists(path):
            print(f"❌ {name}: Model file not found at {path}")
            continue

        print(f"🔍 Evaluating {name}...")
        torch.cuda.empty_cache()

        try:
            model = get_student_model(name, num_classes).cuda()
            state_dict = torch.load(path, map_location="cuda")
            model.load_state_dict(state_dict)

            auroc = evaluate_student(model, val_loader)
            print(f"✅ {name:<16} => AUROC: {auroc:.4f}\n")
        except Exception as e:
            print(f"❌ {name:<16} => Evaluation failed: {e}\n")
