# utils/teacher_eval.py
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from medmnist import INFO
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# -------- CONFIG -------- #
DATA_DIR      = 'data'
MODEL_PATH    = 'models/efficientnet_b3_mnist.pth'
RESULT_DIR    = 'analysis_results'
BATCH_SIZE    = 128
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASETS      = ['pathmnist', 'dermamnist']

os.makedirs(RESULT_DIR, exist_ok=True)

# -------- BUILD TEACHER -------- #
def build_teacher():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    # 1) recreate backbone
    model = models.efficientnet_b3(pretrained=False)
    # remove the classifier
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Identity()
    model.to(DEVICE).eval()
    # load backbone weights
    model.load_state_dict(ckpt['backbone'], strict=False)
    return model, in_feats, ckpt['fc_layers']

# -------- DATA LOADER -------- #
def get_test_loader(ds_name):
    arr = np.load(os.path.join(DATA_DIR, f"{ds_name}.npz"))
    x = arr['test_images']      # (N,H,W,C)
    y = arr['test_labels']      # (N,1)
    # normalize to [0,1] and move channel first
    x = np.transpose(x, (0,3,1,2)).astype(np.float32) / 255.0
    y = y.squeeze().astype(np.int64)

    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

# -------- METRICS & PLOTTING -------- #
def eval_and_plot(model, fc_state, in_feats, ds_name):
    # plug in dataset-specific FC
    n_cls = len(INFO[ds_name]['label'])
    fc = nn.Linear(in_feats, n_cls).to(DEVICE)
    fc.load_state_dict(fc_state)
    fc.eval()

    model.eval()
    all_preds, all_labels = [], []

    loader = get_test_loader(ds_name)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            feats = model(xb)
            logits = fc(feats)
            preds = logits.argmax(1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    # print
    print(f"\n=== {ds_name.upper()} TEACHER METRICS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 score : {f1:.4f}")

    # plot confusion matrix
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Teacher Confusion Matrix\n{ds_name.upper()}")
    plt.colorbar()
    ticks = np.arange(len(INFO[ds_name]['label']))
    plt.xticks(ticks, INFO[ds_name]['label'], rotation=45, ha='right')
    plt.yticks(ticks, INFO[ds_name]['label'])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()

    png_path = os.path.join(
        RESULT_DIR,
        f"teacher_{ds_name}_confusion_matrix.png"
    )
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"📈 Saved plot to {png_path}")

def main():
    model, in_feats, fc_layers = build_teacher()
    for ds in DATASETS:
        eval_and_plot(model, fc_layers[ds], in_feats, ds)

if __name__ == '__main__':
    main()