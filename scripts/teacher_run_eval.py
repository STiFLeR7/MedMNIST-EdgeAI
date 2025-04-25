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

# -------- CONFIG -------- #
DATASETS = ['pathmnist', 'dermamnist']
SAVE_MODEL_PATH = 'models/efficientnet_b3_teacher.pth'
RESULTS_DIR = 'analysis_results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
LR_STEP_SIZE = 10
GAMMA = 0.7

# -------- SETUP DIR -------- #
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------- TRANSFORMS -------- #
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -------- INIT MODEL -------- #
def create_efficientnet_b3():
    model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Identity()
    return model.to(DEVICE)

# -------- DATA LOADER -------- #
def get_loader(dataset_name):
    info = INFO[dataset_name]
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    train_set = DataClass(split='train', root='data', transform=transform, download=True)
    val_set = DataClass(split='val', root='data', transform=transform, download=True)

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, n_classes

# -------- TRAIN -------- #
def train(model, fc_layer, train_loader, criterion, optimizer, scheduler, dataset_name):
    model.train()
    model.classifier[1] = fc_layer

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct, total = 0, 0

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
        wandb.log({f"{dataset_name}/loss": running_loss, f"{dataset_name}/train_acc": acc, "epoch": epoch+1})
        print(f"📘 Epoch {epoch+1}, Loss: {running_loss:.4f}, Train Acc: {acc:.2f}%")

        scheduler.step()

# -------- EVALUATE -------- #
def evaluate(model, fc_layer, val_loader, dataset_name):
    model.eval()
    model.classifier[1] = fc_layer
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

    # Save to file
    result_file = os.path.join(RESULTS_DIR, f"{dataset_name}_accuracy.txt")
    with open(result_file, 'w') as f:
        f.write(f"Validation Accuracy on {dataset_name.upper()}: {acc:.2f}%\n")

# -------- MAIN -------- #
if __name__ == "__main__":
    wandb.login()
    wandb.init(project="medmnist-multitask", name="efficientnet_b3_teacher", config={
        "epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "architecture": "EfficientNet-B3"
    })

    model = create_efficientnet_b3()
    fc_dict = {}

    for dataset_name in DATASETS:
        print(f"\n🔁 Training on {dataset_name.upper()}...")
        train_loader, val_loader, n_classes = get_loader(dataset_name)

        fc_layer = nn.Linear(1536, n_classes).to(DEVICE)
        fc_dict[dataset_name] = fc_layer

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=GAMMA)

        train(model, fc_layer, train_loader, criterion, optimizer, scheduler, dataset_name)

    # Save model backbone + FCs
    model.classifier[1] = nn.Identity()
    torch.save({
        'backbone': model.state_dict(),
        'fc_layers': {k: v.state_dict() for k, v in fc_dict.items()}
    }, SAVE_MODEL_PATH)
    print(f"\n✅ Final model saved to: {SAVE_MODEL_PATH}")

    # -------- POST-TRAINING EVALUATION -------- #
    print("\n🔍 Running Evaluation on all datasets...\n")
    for dataset_name in DATASETS:
        _, val_loader, n_classes = get_loader(dataset_name)
        fc_layer = nn.Linear(1536, n_classes).to(DEVICE)
        fc_layer.load_state_dict(fc_dict[dataset_name].state_dict())
        evaluate(model, fc_layer, val_loader, dataset_name)

    wandb.finish()
