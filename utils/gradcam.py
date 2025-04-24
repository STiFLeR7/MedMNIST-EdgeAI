#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
from medmnist import INFO
import medmnist

# add project root so we can import our model classes from models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mobilenetv3 import MobileNetV3
from models.shufflenetv2 import ShuffleNetV2
from models.tinycnn import TinyCNN

# Directory to save Grad-CAM outputs
gradcam_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gradcam'))
os.makedirs(gradcam_dir, exist_ok=True)

# Normalization stats (mean, std) per dataset
DATASET_MEAN_STD = {
    "pathmnist": ([0.5], [0.5]),
    "dermamnist": ([0.5], [0.5]),
}
# Number of classes per dataset
DATASET_CLASSES = {
    "pathmnist": len(INFO['pathmnist']['label']),
    "dermamnist": len(INFO['dermamnist']['label']),
}

# Base directory for checkpoints
PRETRAINED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))


def get_model(model_type: str, dataset: str):
    """Instantiate a student model and load its checkpoint."""
    n_cls = DATASET_CLASSES[dataset]
    in_ch = INFO[dataset]['n_channels']

    # build model
    if model_type == 'mobilenetv3':
        model = MobileNetV3(num_classes=n_cls)
    elif model_type == 'shufflenetv2':
        model = ShuffleNetV2(num_classes=n_cls)
    elif model_type == 'tinycnn':
        model = TinyCNN(num_classes=n_cls)
        # adapt first conv for in_channels
        model.conv[0] = torch.nn.Conv2d(in_ch, 16, kernel_size=3, padding=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # load checkpoint
    ckpt = os.path.join(PRETRAINED_DIR, f"{model_type}_{dataset}_student.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_sample(dataset: str, idx: int = None):
    """Load a test example from MedMNIST by index or random."""
    mean, std = DATASET_MEAN_STD[dataset]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # use MedMNIST API directly
    info = INFO[dataset]
    DataClass = getattr(medmnist, info['python_class'])
    ds = DataClass(split='test', root='data', transform=transform, download=False)
    if idx is None:
        idx = np.random.randint(len(ds))
    img, _ = ds[idx]
    return img, idx


def apply_gradcam(model: torch.nn.Module, img_tensor: torch.Tensor):
    """Compute Grad-CAM heatmap."""
    grads, acts = [], []

    def fw_hook(m, inp, out): acts.append(out)
    def bw_hook(m, grad_in, grad_out): grads.append(grad_out[0])

    # find last conv layer
    target = next((m for m in reversed(list(model.modules())) if isinstance(m, torch.nn.Conv2d)), None)
    if target is None:
        raise RuntimeError("No Conv2d layer found in model")

    fh = target.register_forward_hook(fw_hook)
    bh = target.register_backward_hook(bw_hook)

    inp = img_tensor.unsqueeze(0)
    out = model(inp)
    pred = out.argmax(dim=1).item()
    loss = out[0, pred]
    model.zero_grad()
    loss.backward()

    grad = grads[0][0].cpu().detach().numpy()
    fmap = acts[0][0].cpu().detach().numpy()
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights): cam += w * fmap[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[2], img_tensor.shape[1]))
    cam -= cam.min(); cam /= cam.max()

    fh.remove(); bh.remove()
    return cam, pred


def save_and_show(img: torch.Tensor, cam: np.ndarray, model_type: str, dataset: str, pred: int, idx: int):
    """Overlay, save and optionally display the Grad-CAM."""
    img_np = img.permute(1, 2, 0).cpu().numpy()
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255
    overlay = 0.5 * heatmap + 0.5 * img_np
    overlay = np.clip(overlay, 0, 1)

    # save to disk
    out_path = os.path.join(gradcam_dir, f"{model_type}_{dataset}_idx{idx}_pred{pred}.png")
    cv2.imwrite(out_path, (overlay * 255).astype(np.uint8))
    print(f"Saved Grad-CAM to {out_path}")

    # display
    plt.figure(figsize=(4,4)); plt.imshow(overlay); plt.axis('off'); plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="GradCAM on MedMNIST student models")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--pathmnist', action='store_true')
    grp.add_argument('--dermamnist', action='store_true')
    grp_mt = parser.add_mutually_exclusive_group(required=True)
    grp_mt.add_argument('--mobilenetv3', action='store_true')
    grp_mt.add_argument('--shufflenetv2', action='store_true')
    grp_mt.add_argument('--tinycnn', action='store_true')
    parser.add_argument('--all', action='store_true', help='Generate for all combinations')
    parser.add_argument('--idx', type=int, help='Index of sample in test set')
    args = parser.parse_args()

    models_list = ['mobilenetv3','shufflenetv2','tinycnn']
    dsets = ['pathmnist','dermamnist']

    def run_one(mt, ds, idx=None):
        print(f"Processing {mt} on {ds}...")
        model = get_model(mt, ds)
        img, i = load_sample(ds, idx)
        cam, pred = apply_gradcam(model, img)
        save_and_show(img, cam, mt, ds, pred, i)

    if args.all:
        for ds in dsets:
            for mt in models_list:
                run_one(mt, ds, args.idx)
    else:
        dataset = 'pathmnist' if args.pathmnist else 'dermamnist'
        mt = 'mobilenetv3' if args.mobilenetv3 else 'shufflenetv2' if args.shufflenetv2 else 'tinycnn'
        run_one(mt, dataset, args.idx)

if __name__ == '__main__':
    main()