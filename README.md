# MedMNIST-EdgeAI 🚑⚡

**MedMNIST-EdgeAI** is a lightweight yet powerful research pipeline built for real-time medical image analysis on edge devices. Leveraging the [MedMNIST](https://medmnist.com/) dataset suite, this project optimizes deep learning pipelines for low-power GPUs and edge accelerators, making it ideal for embedded healthcare AI systems.

---

## 🚀 Strategic Overview

🔍 **Objective**: Democratize medical AI by optimizing classification models for constrained environments without compromising diagnostic accuracy.

🎯 **Impact Goals**:

* Enable real-time diagnostics at the edge
* Shrink model size without significant accuracy loss (via Knowledge Distillation)
* Advance research at the intersection of Edge Computing + Healthcare AI
* Promote reproducibility and modular extensibility for researchers

---

## 📈 Current Scope (v0.1)

* ✅ Standardized, research-ready project structure
* ✅ Teacher model training on 5 MedMNIST datasets using ResNet50
* ✅ Student model distillation using ResNet18, MobileNetV2, EfficientNet-B0
* ✅ Integrated evaluation and performance logging
* ✅ Accuracy and AUROC benchmarking with performance vs size trade-off charts

---

## 📊 Model Performance Snapshot

### 🧠 Teacher Models (ResNet50)

| Dataset     | Metric   | Score |
| ----------- | -------- | ----- |
| PathMNIST   | Accuracy | 0.90  |
| DermaMNIST  | Accuracy | 0.73  |
| OCTMNIST    | Accuracy | 0.92  |
| OrganAMNIST | Accuracy | 0.98  |
| ChestMNIST  | AUROC    | 0.75  |

### 🎓 Student Model Comparison

#### 📌 PathMNIST

| Model           | Accuracy | Params (M) | Reduction (%) | Retained (%) |
| --------------- | -------- | ---------- | ------------- | ------------ |
| ResNet18        | 0.8798   | 11.7       | 54.40         | 97.63        |
| MobileNetV2     | 0.8764   | 4.5        | 86.09         | 97.25        |
| EfficientNet-B0 | 0.8880   | 5.5        | 79.14         | 98.54        |

#### OCTMNIST

| Model           | Accuracy |
| --------------- | -------- |
| ResNet18        | 0.9215   |
| MobileNetV2     | 0.9096   |
| EfficientNet-B0 | 0.8857   |

#### OrganAMNIST

| Model           | Accuracy |
| --------------- | -------- |
| ResNet18        | 0.9754   |
| MobileNetV2     | 0.9672   |
| EfficientNet-B0 | 0.9687   |

#### ChestMNIST (AUROC)

| Model           | AUROC  |
| --------------- | ------ |
| ResNet18        | 0.6926 |
| MobileNetV2     | 0.6692 |
| EfficientNet-B0 | 0.6706 |

#### DermaMNIST

| Model           | Accuracy |
| --------------- | -------- |
| ResNet18        | 0.6949   |
| MobileNetV2     | 0.7019   |
| EfficientNet-B0 | 0.6929   |

---

## 🔧 Tech Stack

* Python 3.10+
* PyTorch
* TorchVision
* MedMNIST
* ONNX / TensorRT (planned)
* NumPy, Matplotlib, Seaborn
* WandB (optional logging)

---

## 🏗️ Project Structure

```bash
MedMNIST-EdgeAI/
├── data/                           # Raw and processed data (gitignored)
├── models/                         # Saved weights and experiments
├── outputs/                        # Performance graphs & evaluation results
├── src/                            # Core source code
│   ├── models/                     # Teacher and student models
│   │   ├── evaluate_all_teachers.py
│   │   ├── evaluate_chest_teacher.py
│   │   ├── teacher_chest.py, etc.
│   │   ├── performance/            # Performance vs size charting
│   │   └── student/                # Distillation and student training
│   │       ├── train_path_student.py, etc.
│   │       ├── student_eval.py
│   ├── utils/                      # Utility configs and tools
│   │   └── config.py
│   ├── dataloader.py              # MedMNIST integration
│   └── loaders.py                 # Dataset loading wrappers
├── download_data.py
├── download_model.py
├── download_student_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧪 Quick Start

```bash
# Install all dependencies
pip install -r requirements.txt

# Train a teacher model (example)
python src/models/train_path_teacher.py --epochs 20

# Evaluate all teacher models
python src/models/evaluate_all_teachers.py

# Train student model (example)
python src/models/student/train_path_student.py --epochs 20

# Evaluate student models
python src/models/student/student_eval.py
```

---

## 🚧 Upcoming Milestones

* 🔲 PyTorch static/dynamic quantization
* 🔲 ONNX export and TensorRT runtime
* 🔲 Jetson Nano & Edge TPU benchmarks
* 🔲 Visual dashboards (ROC, CM, Grad-CAM, etc.)
* 🔲 Android/iOS deployment (TFLite or CoreML)
* 🔲 Edge-LLM integration for clinical support

---

## 🤝 Contributors

* **Stifler** – AI/ML/DL | Edge AI Researcher @ NIMS | CEO, CudaBit

Open to contributors! Fork the repo, raise issues or PRs. Let’s push EdgeAI in healthcare to the next level.

---

## 📜 License

This project is licensed under the **MIT License**.

---

> ⭐️ Star the repo if you find this useful, and follow [Stifler](https://github.com/STiFLeR7) for more edge-optimized research projects.
