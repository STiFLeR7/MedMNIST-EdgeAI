import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from src.loaders import get_dataloaders
from src.models.teacher_template import TeacherResNet50


def get_student_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported student model: {model_name}")

    return model


def distill_student(dataset, student_name, num_classes, teacher_path, student_ckpt_path, epochs=20, alpha=0.5, temperature=4.0):
    print(f"\n📚 Distilling knowledge to {student_name} on {dataset.upper()}...")

    # Load Dataloaders
    data_dir = "data"
    batch_size = 64
    image_size = 28
    num_workers = 0
    train_loader, val_loader, _ = get_dataloaders(data_dir, dataset, batch_size, image_size, num_workers)

    # Load teacher
    teacher = TeacherResNet50(num_classes)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cuda"))
    teacher.cuda().eval()

    # Load student
    student = get_student_model(student_name, num_classes)
    student.load_state_dict(torch.load(f"models/{student_name}/{student_name}.pth", map_location="cpu"), strict=False)
    student.cuda()

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0
    for epoch in range(epochs):
        student.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}]"):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                teacher_outputs = teacher(images)

            student_outputs = student(images)

            loss_kd = criterion_kd(
                nn.functional.log_softmax(student_outputs / temperature, dim=1),
                nn.functional.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature ** 2)

            loss_ce = criterion_ce(student_outputs, labels)
            loss = alpha * loss_kd + (1 - alpha) * loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = student(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), student_ckpt_path)
            print(f"✅ Best model saved to {student_ckpt_path}")

    print(f"🏁 Finished distillation for {student_name} on {dataset}. Best Val Acc: {best_acc:.4f}")
