import os
import torch
import torch.nn as nn
from medmnist import INFO

# -------- CONFIG -------- #
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEVICE     = torch.device('cpu')  # quantization on CPU
DATASETS   = ['pathmnist', 'dermamnist']
MODEL_TYPES= ['mobilenetv3', 'shufflenetv2', 'tinycnn']
MODEL_DIR  = os.path.join(ROOT, 'models')
SAVE_DIR   = os.path.join(ROOT, 'quantization', 'models')
os.makedirs(SAVE_DIR, exist_ok=True)

# -------- STUDENT MODEL FACTORY -------- #
import torchvision.models as models

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
    with torch.no_grad():
        dummy = torch.zeros(1, in_ch, 28, 28)
        flat = nn.Sequential(*layers)(dummy).view(1, -1).size(1)
    seq = layers + [nn.Flatten(), nn.Linear(flat, n_cls)]
    return nn.Sequential(*seq)

# -------- QUANTIZATION UTILS -------- #

def quantize_and_save(model, save_path):
    # dynamic quantize only Linear modules
    q_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(q_model.state_dict(), save_path)
    print(f"Saved quantized model to {save_path}")

# -------- MAIN -------- #
if __name__ == '__main__':
    for ds in DATASETS:
        info  = INFO[ds]
        in_ch = info['n_channels']
        n_cls = len(info['label'])
        for mt in MODEL_TYPES:
            ckpt = os.path.join(MODEL_DIR, f"{mt}_{ds}_student.pth")
            if not os.path.exists(ckpt):
                print(f"✖ Missing checkpoint: {ckpt}")
                continue
            print(f"Quantizing {mt} on {ds}...")
            model = get_student_model(mt, in_ch, n_cls).to(DEVICE)
            state = torch.load(ckpt, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            model.eval()

            out_name = f"{mt}_{ds}_quantized.pth"
            out_path = os.path.join(SAVE_DIR, out_name)
            quantize_and_save(model, out_path)
