from src.models.teacher_template import train_teacher
train_teacher(dataset='chestmnist', num_classes=14, is_multilabel=True)
