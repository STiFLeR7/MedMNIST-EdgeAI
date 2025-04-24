import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from medmnist import INFO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json

# -------- CONFIG -------- #
DATA_DIR = 'data'
MODEL_DIR = 'models'
QUANTIZED_MODEL_DIR = 'quantization/models'
RESULT_DIR = 'analysis_results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
DATASET_NAMES = ['pathmnist', 'dermamnist']
MODELS = ['mobilenetv3', 'shufflenetv2', 'tinycnn']

os.makedirs(RESULT_DIR, exist_ok=True)

# -------- STUDENT MODEL FACTORY (must match training) -------- #
def get_student_model(model_type, in_ch, n_cls):
    if model_type == 'mobilenetv3':
        m = models.mobilenet_v3_small(pretrained=False)
        m.features[0][0] = nn.Conv2d(in_ch, 16, 3, 2, 1, bias=False)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)
        return m
    if model_type == 'shufflenetv2':
        m = models.shufflenet_v2_x1_0(pretrained=False)
        m.conv1[0] = nn.Conv2d(in_ch, 24, 3, 2, 1, bias=False)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
        return m
    # tinycnn
    layers = [
        nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    ]
    m = nn.Sequential(*layers)
    # compute flatten size
    with torch.no_grad():
        dummy = torch.zeros(1, in_ch, 28, 28)
        flat = m(dummy).view(1, -1).size(1)
    m = nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(flat, n_cls)
    )
    return m

# -------- METRICS FUNCTION -------- #
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm.tolist()}

# -------- MAIN ANALYSIS -------- #
def main():
    summary = {}
    for ds in DATASET_NAMES:
        npz = np.load(os.path.join(DATA_DIR, f"{ds}.npz"))
        x_test = npz['test_images']  # (N,H,W,C)
        y_test = npz['test_labels']  # (N,1)
        # reshape and normalize
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32) / 255.0
        y_test = y_test.squeeze().astype(np.int64)
        test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        info = INFO[ds]
        in_ch = info['n_channels']
        n_cls = len(info['label'])

        for mt in MODELS:
            # Main model (non-quantized)
            main_model_path = os.path.join(MODEL_DIR, f"{mt}_{ds}_student.pth")
            print(f"Evaluating {mt} on {ds} (Main Model)...", flush=True)
            model = get_student_model(mt, in_ch, n_cls).to(DEVICE)
            try:
                state = torch.load(main_model_path, map_location=DEVICE)
                model.load_state_dict(state, strict=False)
                metrics = evaluate_model(model, test_loader)
                summary[f"{mt}_{ds}_student"] = metrics
            except Exception as e:
                print(f"❌ Failed to load main model {mt} on {ds}: {e}")

            # Quantized model
            quantized_model_path = os.path.join(QUANTIZED_MODEL_DIR, f"{mt}_{ds}_quantized.pth")
            print(f"Evaluating {mt} on {ds} (Quantized Model)...", flush=True)
            quantized_model = get_student_model(mt, in_ch, n_cls).to(DEVICE)
            try:
                state = torch.load(quantized_model_path, map_location=DEVICE)
                quantized_model.load_state_dict(state, strict=False)
                quantized_metrics = evaluate_model(quantized_model, test_loader)
                summary[f"{mt}_{ds}_quantized"] = quantized_metrics
            except Exception as e:
                print(f"❌ Failed to load quantized model {mt} on {ds}: {e}")

    # Save the summary to a JSON file
    with open(os.path.join(RESULT_DIR, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved metrics_summary.json in", RESULT_DIR)

if __name__ == '__main__':
    main()
