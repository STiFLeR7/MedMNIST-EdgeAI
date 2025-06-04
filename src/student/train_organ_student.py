from src.student.student_template import distill_student

if __name__ == "__main__":
    # Train student: ResNet18 on organaMNIST
    distill_student(
        dataset='organamnist',
        student_name='resnet18',
        num_classes=11,
        teacher_path='models/resnet50_teacher_organamnist.pth',
        student_ckpt_path='models/resnet18/resnet18_organamnist_student.pth'
    )

    # Train student: MobileNetV2 on organaMNIST
    distill_student(
        dataset='organamnist',
        student_name='mobilenet_v2',
        num_classes=11,
        teacher_path='models/resnet50_teacher_organamnist.pth',
        student_ckpt_path='models/mobilenet_v2/mobilenet_v2_organamnist_student.pth'
    )

    # Train student: EfficientNet-B0 on organaMNIST
    distill_student(
        dataset='organamnist',
        student_name='efficientnet_b0',
        num_classes=11,
        teacher_path='models/resnet50_teacher_organamnist.pth',
        student_ckpt_path='models/efficientnet_b0/efficientnet_b0_organamnist_student.pth'
    )
