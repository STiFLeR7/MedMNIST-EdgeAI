#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2

# add scripts folder to path so we can import the student model factory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from train_student import get_student_model

# Dataset normalization and class counts (matching train_student.py)
DATASET_MEAN_STD = {
    "pathmnist": ([0.5], [0.5]),
    "dermamnist": ([0.5], [0.5]),
}
DATASET_CLASSES = {
    "pathmnist": 9,
    "dermamnist": 7,
}


def get_model(model_type: str, dataset: str):
    """Instantiate a student model and load its checkpoint."""
    in_ch = 1  # MedMNIST images are single-channel
    n_cls = DATASET_CLASSES[dataset]
    model = get_student_model(model_type, in_ch, n_cls)

    ckpt = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'models', f"{model_type}_{dataset}_student.pth")
    )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=torch.device('cpu'))
    # if saved as { 'state_dict': {...} }
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # load with strict=False to skip any mismatch
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_sample(dataset: str):
    """Load a random test image from MedMNIST folder structure."""
    mean, std = DATASET_MEAN_STD[dataset]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    root = os.path.join('data', dataset)
    test_ds = datasets.ImageFolder(root=os.path.join(root, 'test'), transform=transform)
    loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    img, _ = next(iter(loader))
    return img[0]


def apply_gradcam(model: torch.nn.Module, img_tensor: torch.Tensor):
    """Compute Grad-CAM heatmap for a single input."""
    grads, acts = [], []

    def fw_hook(m, inp, out): acts.append(out)
    def bw_hook(m, grad_in, grad_out): grads.append(grad_out[0])

    # find last Conv2d layer
    target = None
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            target = m
            break
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
    weights = np.mean(grad, axis=(1,2))

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights): cam += w * fmap[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[2], img_tensor.shape[1]))
    cam -= cam.min(); cam /= cam.max()

    fh.remove(); bh.remove()
    return cam


def overlay_and_show(img: torch.Tensor, cam: np.ndarray):
    """Overlay heatmap on image and display."""
    img_np = img.permute(1,2,0).cpu().numpy()
    heatmap = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32)/255
    overlay = 0.5*heatmap + 0.5*img_np
    overlay = np.clip(overlay, 0, 1)
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="GradCAM on MedMNIST student models")
    grp_ds = parser.add_mutually_exclusive_group(required=True)
    grp_ds.add_argument('--pathmnist', action='store_true', help='Use PathMNIST')
    grp_ds.add_argument('--dermamnist', action='store_true', help='Use DermaMNIST')
    grp_mt = parser.add_mutually_exclusive_group(required=True)
    grp_mt.add_argument('--mobilenetv3', action='store_true', help='MobileNetV3 small')
    grp_mt.add_argument('--shufflenetv2', action='store_true', help='ShuffleNetV2')
    grp_mt.add_argument('--tinycnn', action='store_true', help='TinyCNN')
    args = parser.parse_args()

    dataset = 'pathmnist' if args.pathmnist else 'dermamnist'
    if args.mobilenetv3:
        model_type = 'mobilenetv3'
    elif args.shufflenetv2:
        model_type = 'shufflenetv2'
    else:
        model_type = 'tinycnn'

    print(f"Loading {model_type} on {dataset}…")
    model = get_model(model_type, dataset)
    sample = load_sample(dataset)
    cam = apply_gradcam(model, sample)
    overlay_and_show(sample, cam)


if __name__=='__main__':
    main()