# utils/student_qat_run_eval.py

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
from torch.optim.lr_scheduler import StepLR

# ─── CONFIG ───────────────────────────────────────────────────────────────────────
DATASETS     = ['pathmnist','dermamnist']
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STUDENT_DIR  = 'models/student_models'
TEACHER_CKPT = 'models/efficientnet_b3_teacher.pth'
os.makedirs(STUDENT_DIR, exist_ok=True)
os.makedirs('analysis_results', exist_ok=True)

BATCH_SIZE    = 32
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-3
LR_STEP_SIZE  = 10
GAMMA         = 0.7
ALPHA         = 0.5    # distillation weight
TEMPERATURE   = 4.0

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ─── TINYCNN ────────────────────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

# ─── STUDENT FACTORY (NO DOWNLOAD!) ────────────────────────────────────────────────
def get_student_model(name, ncls):
    if name=='mobilenetv3':
        m = models.mobilenet_v3_small(pretrained=False)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, ncls)
    elif name=='shufflenet':
        m = models.shufflenet_v2_x1_0(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, ncls)
    elif name=='tinycnn':
        m = TinyCNN(ncls)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m.to(DEVICE)

# ─── LOAD ALREADY‑TRAINED STUDENT ─────────────────────────────────────────────────
def load_student(name, ds, ncls):
    student = get_student_model(name, ncls)
    ckpt_path = os.path.join(STUDENT_DIR, f"{name}_{ds}.pth")
    state = torch.load(ckpt_path, map_location=DEVICE)
    student.load_state_dict(state)
    return student

# ─── DATALOADER ──────────────────────────────────────────────────────────────────
def get_loader(ds):
    info = INFO[ds]
    DataClass = getattr(medmnist, info['python_class'])
    tr = DataClass(split='train',root='data',transform=transform,download=True)
    va = DataClass(split='val',  root='data',transform=transform,download=True)
    return (
      DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True),
      DataLoader(va,batch_size=BATCH_SIZE,shuffle=False),
      len(info['label'])
    )

# ─── MULTI‑TASK TEACHER ────────────────────────────────────────────────────────────
def load_teacher(ds, ncls):
    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE)
    # backbone
    backbone = models.efficientnet_b3(pretrained=False)
    in_feats = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(ckpt['backbone'], strict=False)
    backbone.eval().to(DEVICE)
    # head
    head = nn.Linear(in_feats, ncls).to(DEVICE)
    head.load_state_dict(ckpt['fc_layers'][ds], strict=False)
    head.eval()
    return nn.Sequential(backbone, head).eval()

# ─── DISTILLATION LOSS ────────────────────────────────────────────────────────────
def distillation_loss(s_logits, t_logits, targets):
    kd = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(s_logits/TEMPERATURE, dim=1),
        nn.functional.softmax(t_logits/TEMPERATURE, dim=1)
    ) * (TEMPERATURE**2)
    ce = nn.CrossEntropyLoss()(s_logits, targets)
    return ALPHA*kd + (1-ALPHA)*ce

# ─── TRAIN + QAT (FP32) ────────────────────────────────────────────────────────────
def train_qat(student, teacher, tr_loader, ds, name):
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    sch = StepLR(opt, step_size=LR_STEP_SIZE, gamma=GAMMA)

    for ep in range(NUM_EPOCHS):
        running, corr, tot = 0., 0, 0
        for x,y in tqdm(tr_loader, desc=f"[{name}][{ds}] Ep{ep+1}/{NUM_EPOCHS}"):
            x,y = x.to(DEVICE), y.squeeze().long().to(DEVICE)
            with torch.no_grad():
                tlog = teacher(x)
            slog = student(x)
            loss = distillation_loss(slog, tlog, y)

            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
            preds = slog.argmax(1)
            corr += (preds==y).sum().item(); tot += y.size(0)

        wandb.log({f"{ds}/{name}/loss": running,
                   f"{ds}/{name}/acc": 100*corr/tot,
                   "epoch": ep+1})
        print(f" → Loss {running:.4f} | Acc {100*corr/tot:.2f}%")
        sch.step()

    # ─── DYNAMIC QUANTIZE ONLY FC LAYERS ───────────────────────────────────────────
    student.cpu().eval()
    quant = torch.quantization.quantize_dynamic(
        student, {nn.Linear}, dtype=torch.qint8
    )
    out_path = os.path.join(STUDENT_DIR, f"{name}_{ds}_qat.pth")
    torch.save(quant.state_dict(), out_path)
    print(f"💾 Quantized student saved: {out_path}")
    return quant

# ─── CPU‑ONLY EVALUATION ──────────────────────────────────────────────────────────
def evaluate(student, val_loader, ds, name):
    student.eval().cpu()
    corr, tot = 0,0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.cpu(), y.squeeze().long().cpu()
            out = student(x)
            preds = out.argmax(1)
            corr += (preds==y).sum().item(); tot += y.size(0)

    acc = 100*corr/tot
    print(f"🧪 [{name}][{ds}] QAT Val Acc: {acc:.2f}%")
    wandb.log({f"{ds}/{name}/qat_val_acc": acc})
    with open(f"analysis_results/{name}_{ds}_qat_eval.txt","w") as f:
        f.write(f"{name} on {ds} QAT Acc: {acc:.2f}%\n")

# ─── MAIN ─────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    for student_name in ['mobilenetv3','shufflenet','tinycnn']:
      for ds in DATASETS:
        print(f"\n🔥 QAT+Distill {student_name} on {ds}")
        wandb.init(
          project="medmnist-qat-distill",
          name=f"{student_name}_{ds}_qat",
          config={"epochs":NUM_EPOCHS,"lr":LEARNING_RATE,
                  "batch_size":BATCH_SIZE,
                  "alpha":ALPHA, "T":TEMPERATURE}
        )

        tr_ld, val_ld, ncls = get_loader(ds)
        # load your **already‑trained** student (no downloads!):
        student = load_student(student_name, ds, ncls)
        teacher = load_teacher(ds, ncls)

        quant_student = train_qat(student, teacher, tr_ld, ds, student_name)
        evaluate(quant_student, val_ld, ds, student_name)

        wandb.finish()
