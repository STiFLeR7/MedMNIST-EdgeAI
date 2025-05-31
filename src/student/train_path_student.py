import os
from src.student.student_template import distill_student

if __name__ == "__main__":
    dataset = "pathmnist"
    teacher_ckpt = "models/resnet50_teacher_pathmnist.pth"

    student_configs = [
        {
            "name": "resnet18",
            "ckpt_path": "models/resnet18/resnet18.pth"
        },
        {
            "name": "mobilenet_v2",
            "ckpt_path": "models/mobilenet_v2/mobilenet_v2.pth"
        },
        {
            "name": "efficientnet_b0",
            "ckpt_path": "models/efficientnet_b0/efficientnet_b0.pth"
        }
    ]

    for student in student_configs:
        print(f"\n🚀 Starting distillation: {student['name']} on {dataset}...")
        distill_student(
            dataset=dataset,
            teacher_ckpt=teacher_ckpt,
            student_name=student['name'],
            student_ckpt=student['ckpt_path']
        )
