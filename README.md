# MedMNIST-EdgeAI 🚑⚡

**MedMNIST-EdgeAI** is an edge-optimized deep learning pipeline focused on medical image classification using the [MedMNIST](https://medmnist.com/) dataset collection. This project emphasizes real-time performance, low-latency inference, and efficient deployment on resource-constrained devices, including mobile GPUs and edge accelerators.

## 📌 Current Scope (v0.1)

* ✅ Standardized project structure for scalability and modularity.
* ✅ Integration of MedMNIST datasets using `medmnist` library.
* ✅ Baseline training pipeline using CNNs and basic augmentation techniques.
* ✅ Logging setup for clean experiment tracking.
* ✅ `.gitignore` configured to exclude data and Python caches.
* ✅ **Teacher model training complete** for 5 MedMNIST datasets:

| Dataset     | Metric   | Score  |
| ----------- | -------- | ------ |
| PathMNIST   | Accuracy | \~0.90 |
| ChestMNIST  | AUROC    | \~0.75 |
| OrganAMNIST | Accuracy | \~0.98 |
| DermaMNIST  | Accuracy | \~0.73 |
| OCTMNIST    | Accuracy | \~0.92 |

## 🚧 Under Development

* ⏳ Custom model experimentation (e.g., MobileNetV3, EfficientNet-lite, etc.).
* ⏳ Torch quantization pipeline (static/dynamic).
* ⏳ ONNX/TensorRT conversion for inference benchmarking.
* ⏳ Benchmarking on RTX 3050 vs. Jetson/Edge TPU.

## 🔮 Future Roadmap

* 🧠 Edge LLM integration for real-time decision support (Doctor AI Assistant).
* 🚁 Federated learning pipeline for privacy-preserving medical AI.
* 📲 Android/iOS inference app for real-time diagnostics.
* 📊 Integrated visual dashboard for predictions, ROC, confusion matrix, and misclassification analysis.
* 🤝 Collaboration with healthcare domain experts for use-case validation.

## 🧱 Tech Stack

* Python 3.10+
* PyTorch
* MedMNIST
* TorchVision
* NumPy, Matplotlib, Seaborn
* ONNX / TensorRT (planned)
* WandB (optional logging)

## 📂 Project Structure

```bash
MedMNIST-EdgeAI/
├── src/                  # All source code
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model definitions
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Evaluation script
│   └── utils.py          # Utility functions
├── data/                 # (gitignored) Raw and processed data
├── experiments/          # Saved models, logs, checkpoints
├── README.md             # This file
└── requirements.txt      # Environment dependencies
```

## 🧪 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py --dataset pathoMNIST --epochs 20

# Evaluate model
python src/evaluate.py --model checkpoints/best_model.pth
```

## 🤝 Contributors

* **Stifler** – Researcher & Developer @ NIMS | AI/ML/DL | CudaBit Tech Lead
* Open to contributions! If you're passionate about AI + Healthcare + Edge, ping me.

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more details.
