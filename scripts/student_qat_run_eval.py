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
from torch.ao.quantization import prepare_qat, convert

os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'models', 'student_models')

# -------- CONFIG -------- #
DATASETS = ['pathmnist', 'dermamnist']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'models/student_models/'
TEACHER_PATHS = {
    'pathmnist': 'models/efficientnet_b3_teacher_pathmnist.pth',
    'dermamnist': 'models/efficientnet_b3_teacher_dermamnist.pth'
}
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
LR_STEP_SIZE = 10
GAMMA = 0.7
ALPHA = 0.5  # distillation loss weight
TEMPERATURE = 4.0

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

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

def get_loader(dataset_name):
    info = INFO[dataset_name]
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    train_set = DataClass(split='train', root='data', transform=transform, download=True)
    val_set = DataClass(split='val', root='data', transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, n_classes

def load_teacher(dataset_name, num_classes):
    teacher = models.efficientnet_b3(weights=None)
    teacher.classifier[1] = nn.Linear(teacher.classifier[1].in_features, num_classes)
    teacher.load_state_dict(torch.load(TEACHER_PATHS[dataset_name], map_location=DEVICE))
    return teacher.to(DEVICE).eval()

def distillation_loss(student_logits, teacher_logits, targets, temperature, alpha):
    kd_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_logits / temperature, dim=1),
                                                   nn.functional.softmax(teacher_logits / temperature, dim=1)) * (temperature ** 2)
    ce_loss = nn.CrossEntropyLoss()(student_logits, targets)
    return alpha * kd_loss + (1 - alpha) * ce_loss

def train_qat(model, teacher, train_loader, val_loader, dataset_name, model_name):
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    prepare_qat(model, inplace=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=GAMMA)

    for epoch in range(NUM_EPOCHS):
        total_loss, correct, total = 0, 0, 0
        for images, targets in tqdm(train_loader, desc=f"[{model_name.upper()}] Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, targets = images.to(DEVICE), targets.squeeze().long().to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(images)

            outputs = model(images)
            loss = distillation_loss(outputs, teacher_logits, targets, TEMPERATURE, ALPHA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        acc = 100 * correct / total
        print(f"📘 Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {acc:.2f}%")
        wandb.log({f"{dataset_name}/{model_name}/qat_loss": total_loss, f"{dataset_name}/{model_name}/qat_acc": acc, "epoch": epoch + 1})
        scheduler.step()

    model.cpu()
    convert(model.eval(), inplace=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{model_name}_{dataset_name}_qat.pth"))
    print(f"💾 Quantized model saved: {model_name}_{dataset_name}_qat.pth")

def evaluate(model, val_loader, dataset_name, model_name):
    model.eval()
    model.to(DEVICE)
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(DEVICE), targets.squeeze().long().to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    acc = 100 * correct / total
    print(f"🧪 {dataset_name.upper()} - {model_name} QAT Val Accuracy: {acc:.2f}%")
    with open(f'analysis_results/{model_name}_{dataset_name}_qat_eval.txt', 'w') as f:
        f.write(f"{dataset_name} {model_name} QAT Accuracy: {acc:.2f}%\n")
    wandb.log({f"{dataset_name}/{model_name}/qat_val_acc": acc})

if __name__ == "__main__":
    os.makedirs("analysis_results", exist_ok=True)
    student_models = ['mobilenetv3', 'shufflenet', 'tinycnn']

    for student_name in student_models:
        for dataset_name in DATASETS:
            print(f"\n🔥 Starting QAT + Distillation for {student_name} on {dataset_name}")
            wandb.init(project="medmnist-qat-distill", name=f"{student_name}_{dataset_name}_qat", config={
                "epochs": NUM_EPOCHS,
                "lr": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "student_arch": student_name,
                "dataset": dataset_name,
                "temperature": TEMPERATURE,
                "alpha": ALPHA
            })

            train_loader, val_loader, num_classes = get_loader(dataset_name)
            student = get_student_model(student_name, num_classes)
            teacher = load_teacher(dataset_name, num_classes)
            train_qat(student, teacher, train_loader, val_loader, dataset_name, student_name)
            evaluate(student, val_loader, dataset_name, student_name)

            wandb.finish()
