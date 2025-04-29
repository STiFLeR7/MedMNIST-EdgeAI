import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from medmnist import INFO
import medmnist
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import StepLR
import json

# -------- CONFIG -------- #
SAVE_DIR = 'models/'
ANALYSIS_DIR = 'analysis_results/'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
LR_STEP_SIZE = 10
GAMMA = 0.7

# -------- TRANSFORMS -------- #
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -------- FUNCTIONS -------- #

def create_efficientnet_b3(n_classes):
    model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes).to(DEVICE)
    return model.to(DEVICE)

def get_loader(dataset_name):
    info = INFO[dataset_name]
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    train_set = DataClass(split='train', root='data', transform=transform, download=True)
    val_set = DataClass(split='val', root='data', transform=transform, download=True)

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, n_classes

def train_and_eval(dataset_name):
    # W&B session
    wandb.init(project="medmnist-multitask", name=f"efficientnet_b3_{dataset_name}", config={
        "dataset": dataset_name,
        "epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "architecture": "EfficientNet-B3"
    })

    print(f"\n🔵 Starting training on {dataset_name.upper()}...")
    train_loader, val_loader, n_classes = get_loader(dataset_name)
    model = create_efficientnet_b3(n_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=GAMMA)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, targets in tqdm(train_loader, desc=f"[{dataset_name.upper()}] Epoch [{epoch+1}/{NUM_EPOCHS}]"):
            images, targets = images.to(DEVICE), targets.squeeze().long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        acc = 100 * correct / total
        wandb.log({f"{dataset_name}/train_loss": running_loss, f"{dataset_name}/train_acc": acc, "epoch": epoch+1})
        print(f"📘 Epoch {epoch+1}, Loss: {running_loss:.4f}, Train Acc: {acc:.2f}%")

        scheduler.step()

    # Save model
    save_path = os.path.join(SAVE_DIR, f"efficientnet_b3_teacher_{dataset_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved model at {save_path}")

    # Evaluation
    eval_acc = evaluate(model, val_loader, dataset_name)
    return eval_acc

def evaluate(model, val_loader, dataset_name):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(DEVICE), targets.squeeze().long().to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    acc = 100 * correct / total
    wandb.log({f"{dataset_name}/val_acc": acc})
    print(f"🧪 {dataset_name.upper()} Validation Accuracy: {acc:.2f}%")

    # Save result
    result = {
        "dataset": dataset_name,
        "validation_accuracy": acc
    }
    with open(os.path.join(ANALYSIS_DIR, f"{dataset_name}_result.json"), 'w') as f:
        json.dump(result, f, indent=4)

    return acc

# -------- MAIN -------- #

if __name__ == "__main__":
    wandb.login()

    datasets = ['pathmnist', 'dermamnist']
    summary = {}

    for dataset_name in datasets:
        acc = train_and_eval(dataset_name)
        summary[dataset_name] = acc
        wandb.finish()

    print("\n📊 Final Summary:")
    for k, v in summary.items():
        print(f"{k.upper()} Validation Accuracy: {v:.2f}%")

