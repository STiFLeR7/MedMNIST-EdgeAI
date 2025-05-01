import os
import torch
import torch.nn as nn
import torch.quantization
import torchvision.models as models
from medmnist import INFO
import medmnist
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 10  # QAT epochs can be fewer
LEARNING_RATE = 1e-4
MODEL_DIR = "models/student_models/"
SAVE_DIR = "models/qat_models/"
os.makedirs(SAVE_DIR, exist_ok=True)

DATASETS = ['pathmnist', 'dermamnist']
MODEL_NAMES = ['mobilenetv3', 'shufflenet', 'tinycnn']

# ---------- Transforms ---------- #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5]*3, std=[.5]*3)
])

# ---------- Model Factory ---------- #
def create_model(name, num_classes):
    if name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == 'shufflenet':
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'tinycnn':
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    return model.to(DEVICE)

# ---------- Data Loader ---------- #
def get_loader(dataset_name):
    info = INFO[dataset_name]
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    train_set = DataClass(split='train', root='data', transform=transform, download=True)
    val_set = DataClass(split='val', root='data', transform=transform, download=True)

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, n_classes

# ---------- QAT Training ---------- #
def train_qat(model, train_loader, val_loader, dataset_name, model_name):
    model.train()
    model.fuse_model() if hasattr(model, 'fuse_model') else None
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        running_loss, correct, total = 0.0, 0, 0
        for images, targets in tqdm(train_loader, desc=f"[{model_name.upper()}] Epoch {epoch+1}/{NUM_EPOCHS}"):
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
        print(f"📘 Epoch {epoch+1} | Loss: {running_loss:.4f} | Acc: {100*correct/total:.2f}%")

    torch.quantization.convert(model.eval(), inplace=True)
    evaluate(model, val_loader, dataset_name, model_name)
    save_path = os.path.join(SAVE_DIR, f"{model_name}_{dataset_name}_qat.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 QAT Model saved: {save_path}")

# ---------- Evaluation ---------- #
def evaluate(model, val_loader, dataset_name, model_name):
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
    print(f"🧪 {model_name.upper()} on {dataset_name.upper()} QAT Accuracy: {acc:.2f}%")

# ---------- Main ---------- #
if __name__ == "__main__":
    for dataset_name in DATASETS:
        train_loader, val_loader, num_classes = get_loader(dataset_name)
        for model_name in MODEL_NAMES:
            model = create_model(model_name, num_classes)
            model_path = os.path.join(MODEL_DIR, f"{model_name}_{dataset_name}.pth")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            train_qat(model, train_loader, val_loader, dataset_name, model_name)
            print(f"🚀 QAT Training completed for {model_name} on {dataset_name}.\n")