import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os
from src.models.teacher_template import TeacherResNet50
from src.loaders import get_dataloaders
import torch.nn.functional as F

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

def distill_student(dataset, student_name, num_classes, teacher_path, student_ckpt_path):
    print(f"\n📚 Distilling knowledge to {student_name} on {dataset.upper()}...")

    # Data
    train_loader, val_loader, _ = get_dataloaders("data", dataset, batch_size=64, image_size=28, num_workers=2)

    # Teacher model
    teacher = TeacherResNet50(num_classes).cuda()
    teacher.load_state_dict(torch.load(teacher_path, map_location="cuda"))
    teacher.eval()

    # Student model
    student = get_student_model(student_name, num_classes).cuda()
    pretrained_path = f"models/{student_name}/{student_name}.pth"

    if os.path.exists(pretrained_path):
        print(f"🔄 Loading base weights for {student_name} from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')

        # Remove classifier/fc layers
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("fc.") and not k.startswith("classifier.")
        }

        missing, unexpected = student.load_state_dict(filtered_state_dict, strict=False)
        print(f"ℹ️ Loaded base weights with missing keys: {missing}")
    else:
        print(f"⚠️ No base weights found for {student_name}, training from scratch.")

    # Losses and optimizer
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student.parameters(), lr=1e-3)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}]"):
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                teacher_outputs = teacher(images)

            student_outputs = student(images)

            loss_ce = criterion_ce(student_outputs, labels)
            loss_kd = criterion_kd(
                F.log_softmax(student_outputs / 4.0, dim=1),
                F.softmax(teacher_outputs / 4.0, dim=1)
            ) * (4.0 ** 2)

            loss = 0.5 * loss_ce + 0.5 * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"✅ Epoch {epoch+1}: Avg Loss = {running_loss / len(train_loader):.4f}")

    # Save final student checkpoint
    torch.save(student.state_dict(), student_ckpt_path)
    print(f"💾 Saved student model to {student_ckpt_path}")
