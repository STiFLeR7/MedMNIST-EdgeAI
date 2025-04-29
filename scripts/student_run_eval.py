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

os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'models', 'student_models')

# -------- CONFIG -------- #
DATASETS = ['pathmnist', 'dermamnist']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'models/student_models/'
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
LR_STEP_SIZE = 10
GAMMA = 0.7

# -------- TRANSFORMS -------- #
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# -------- MODEL DEFINITIONS -------- #
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_student_model(name, num_classes):
    if name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == 'shufflenet':
        model = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'tinycnn':
        model = TinyCNN(num_classes)
    else:
        raise ValueError(f"Unknown student model: {name}")
    return model.to(DEVICE)

# -------- DATA LOADER -------- #
def get_loader(dataset_name):
    info = INFO[dataset_name]
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    train_set = DataClass(split='train', root='data', transform=transform, download=True)
    val_set = DataClass(split='val', root='data', transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, n_classes

# -------- TRAIN -------- #
def train(model, train_loader, optimizer, criterion, scheduler, dataset_name):
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        correct, total = 0, 0

        for images, targets in tqdm(train_loader, desc=f"[{dataset_name}] Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, targets = images.to(DEVICE), targets.squeeze().long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        acc = 100 * correct / total
        wandb.log({f"{dataset_name}/train_loss": total_loss, f"{dataset_name}/train_acc": acc, "epoch": epoch + 1})
        print(f"📘 Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {acc:.2f}%")
        scheduler.step()

# -------- EVALUATE -------- #
def evaluate(model, val_loader, dataset_name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(DEVICE), targets.squeeze().long().to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    acc = 100 * correct / total
    wandb.log({f"{dataset_name}/val_acc": acc})
    print(f"🧪 {dataset_name.upper()} Validation Accuracy: {acc:.2f}%")

    with open(f'analysis_results/{dataset_name}_student_eval.txt', 'w') as f:
        f.write(f"{dataset_name} student model accuracy: {acc:.2f}%\n")

# -------- MAIN -------- #
if __name__ == "__main__":
    os.makedirs("analysis_results", exist_ok=True)
    student_models = ['mobilenetv3', 'shufflenet', 'tinycnn']

    for student_name in student_models:
        wandb.init(project="medmnist-multitask", name=f"{student_name}_student", config={
            "epochs": NUM_EPOCHS,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "architecture": student_name
        })

        for dataset_name in DATASETS:
            print(f"\n🚀 Training {student_name} on {dataset_name}...")
            train_loader, val_loader, num_classes = get_loader(dataset_name)
            model = get_student_model(student_name, num_classes)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=GAMMA)
            criterion = nn.CrossEntropyLoss()

            train(model, train_loader, optimizer, criterion, scheduler, dataset_name)
            evaluate(model, val_loader, dataset_name)

            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{student_name}_{dataset_name}.pth"))
            print(f"💾 Saved model: {student_name}_{dataset_name}.pth")

        wandb.finish()
