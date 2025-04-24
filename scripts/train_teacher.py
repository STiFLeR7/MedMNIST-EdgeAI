import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from medmnist import INFO
import medmnist
from tqdm import tqdm
import wandb

# -------- CONFIG -------- #
DATASETS = ['pathmnist', 'dermamnist']
SAVE_MODEL_PATH = 'models/efficientnet_b3_mnist.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# -------- TRANSFORMS -------- #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# -------- INIT MODEL -------- #
def create_efficientnet_b3():
    # Load pretrained B3
    model = models.efficientnet_b3(pretrained=True)
    # Grab the original classifier’s in_features
    orig_in_feats = model.classifier[1].in_features
    # Remove the old classifier
    model.classifier = nn.Identity()
    # Stash that in_features for later
    model.backbone_out = orig_in_feats
    return model.to(DEVICE)

# -------- DATA LOADER -------- #
def get_loader(dataset_name):
    info = INFO[dataset_name]
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    train_set = DataClass(split='train', root='data', transform=transform, download=False)
    val_set   = DataClass(split='val',   root='data', transform=transform, download=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, n_classes

# -------- TRAINING -------- #
def train(model, fc_layer, loader, criterion, optimizer, name):
    model.train()
    model.classifier = fc_layer

    for ep in range(NUM_EPOCHS):
        running_loss = correct = total = 0
        pbar = tqdm(loader, desc=f"[{name.upper()}] Epoch {ep+1}/{NUM_EPOCHS}")
        for imgs, tgts in pbar:
            imgs, tgts = imgs.to(DEVICE), tgts.squeeze().long().to(DEVICE)

            outs = model(imgs)
            loss = criterion(outs, tgts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outs.argmax(1)
            correct   += (preds==tgts).sum().item()
            total     += tgts.size(0)

        acc = 100*correct/total
        wandb.log({f"{name}/loss": running_loss,
                   f"{name}/train_acc": acc,
                   "epoch": ep+1})
        print(f" → Loss {running_loss:.4f} | Acc {acc:.2f}%")

# -------- VALIDATION -------- #
def evaluate(model, fc_layer, loader, name):
    model.eval()
    model.classifier = fc_layer
    correct = total = 0
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs, tgts = imgs.to(DEVICE), tgts.squeeze().long().to(DEVICE)
            outs = model(imgs)
            preds = outs.argmax(1)
            correct += (preds==tgts).sum().item()
            total   += tgts.size(0)
    acc = 100*correct/total
    wandb.log({f"{name}/val_acc": acc})
    print(f" 🧪 {name.upper()} Val Acc: {acc:.2f}%")

# -------- MAIN -------- #
if __name__ == "__main__":
    wandb.login()
    wandb.init(project="medmnist-multitask",
               name="efficientnet_b3_teacher",
               config={"epochs": NUM_EPOCHS, "lr": LEARNING_RATE,
                       "batch_size": BATCH_SIZE, "arch": "EfficientNet-B3"})

    model = create_efficientnet_b3()
    fc_dict = {}

    for ds in DATASETS:
        print(f"\n🔁 Training on {ds.upper()} …")
        tr_ld, va_ld, n_cls = get_loader(ds)
        # build a fresh fc layer per dataset
        fc = nn.Linear(model.backbone_out, n_cls).to(DEVICE)
        fc_dict[ds] = fc

        opt = torch.optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=LEARNING_RATE)
        crit = nn.CrossEntropyLoss()

        train(model, fc, tr_ld, crit, opt, ds)
        evaluate(model, fc, va_ld, ds)

    # Save backbone & all FCs
    torch.save({
        'backbone': model.state_dict(),
        'fc_layers': {k: v.state_dict() for k, v in fc_dict.items()}
    }, SAVE_MODEL_PATH)
    print(f"\n✅ Saved full EfficientNet-B3 teacher at {SAVE_MODEL_PATH}")
    wandb.finish()
