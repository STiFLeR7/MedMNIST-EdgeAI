from src.student.student_template import distill_student

# Train student: ResNet18 on PathMNIST
distill_student(
    dataset='pathmnist',
    student_name='resnet18',
    num_classes=9,
    teacher_path='models/resnet50_teacher_pathmnist.pth',
    student_ckpt_path='models/resnet18/resnet18_pathmnist_student.pth'
)

# Train student: MobileNetV2 on PathMNIST
distill_student(
    dataset='pathmnist',
    student_name='mobilenet_v2',
    num_classes=9,
    teacher_path='models/resnet50_teacher_pathmnist.pth',
    student_ckpt_path='models/mobilenet_v2/mobilenet_v2_pathmnist_student.pth'
)

# Train student: EfficientNet-B0 on PathMNIST
distill_student(
    dataset='pathmnist',
    student_name='efficientnet_b0',
    num_classes=9,
    teacher_path='models/resnet50_teacher_pathmnist.pth',
    student_ckpt_path='models/efficientnet_b0/efficientnet_b0_pathmnist_student.pth'
)
