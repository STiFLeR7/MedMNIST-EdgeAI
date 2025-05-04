
import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat_fx, convert_fx
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import torch.backends.cudnn as cudnn

from models.student_models import get_student_model
from models.teacher_model import get_teacher_model
from medmnist import INFO, Evaluator
from medmnist.dataset import PathMNIST, DermaMNIST

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 10
FINETUNE_EPOCHS = 3
DATA_PATH = "./data"

def get_dataset(name):
    info = INFO[name]
    DataClass = eval(info['python_class'])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    train_dataset = DataClass(split='train', transform=transform, download=True, root=DATA_PATH)
    val_dataset = DataClass(split='val', transform=transform, download=True, root=DATA_PATH)
    return train_dataset, val_dataset, info['n_classes']

def load_model(path, model):
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model

def train_one_epoch(model, teacher, loader, optimizer, criterion, epoch, is_qat=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in tqdm(loader, desc=f"[Ep{epoch+1}] QAT:{is_qat}", leave=False):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        if teacher:
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            loss = criterion(outputs, teacher_outputs)
        else:
            loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100. * correct / total

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct / total

def train_qat(student, teacher, train_loader, val_loader):
    student.to(DEVICE)
    teacher.to(DEVICE) if teacher else None

    # QAT Config
    qconfig = get_default_qat_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    example_inputs = torch.randn(1, 3, 28, 28).to(DEVICE)

    # Skip quantization for grouped convolutions by excluding them
    student.eval()
    student_prepared = prepare_qat_fx(student, qconfig_dict, example_inputs)
    student_prepared.train()

    # Optimizer
    optimizer = optim.Adam(student_prepared.parameters(), lr=1e-4)
    criterion = nn.MSELoss() if teacher else nn.CrossEntropyLoss()

    # QAT Training
    for epoch in range(NUM_EPOCHS):
        loss, acc = train_one_epoch(student_prepared, teacher, train_loader, optimizer, criterion, epoch, is_qat=True)
        print(f" → Ep{epoch+1} Loss: {loss:.4f} Acc: {acc:.2f}%")

    # Convert to quantized model
    student_int8 = convert_fx(student_prepared.eval())

    # Finetuning phase
    print("🔧 Finetuning quantized model...")
    student_int8.to(DEVICE)
    optimizer = optim.Adam(student_int8.parameters(), lr=1e-5)
    for epoch in range(FINETUNE_EPOCHS):
        loss, acc = train_one_epoch(student_int8, None, train_loader, optimizer, None, epoch, is_qat=False)
        print(f" [Finetune] Ep{epoch+1} Acc: {acc:.2f}%")

    final_acc = evaluate(student_int8, val_loader)
    return final_acc

if __name__ == "__main__":
    settings = [
        ("mobilenetv3", "pathmnist"),
        ("mobilenetv3", "dermamnist"),
        ("shufflenet", "pathmnist"),
        ("shufflenet", "dermamnist"),
        ("tinycnn", "pathmnist"),
        ("tinycnn", "dermamnist"),
    ]

    for student_name, ds in settings:
        print(f"🔥 QAT+static {student_name} on {ds}")
        train_ds, val_ds, num_classes = get_dataset(ds)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        student_path = f"models/student_models/{student_name}_{ds}.pth"
        teacher_path = "models/teacher_models/efficientnet_b3_teacher.pth"

        student = load_model(student_path, get_student_model(student_name, num_classes))
        teacher = load_model(teacher_path, get_teacher_model(num_classes))

        acc = train_qat(student, teacher, train_loader, val_loader)
        print(f"✅ {student_name} on {ds} QAT+Finetune Acc: {acc:.2f}%\n")
