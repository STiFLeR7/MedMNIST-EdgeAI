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
from torch.ao.quantization import prepare_qat, convert

# ─── Use FBGEMM for quantized CPUs ───────────────────────────────────────────
torch.backends.quantized.engine = 'fbgemm'

# ─── ENV ─────────────────────────────────────────────────────────────────────────
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'models', 'student_models')

# ─── CONFIG ───────────────────────────────────────────────────────────────────────
DATASETS    = ['pathmnist','dermamnist']
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR    = 'models/student_models/'
TEACHER_CKPT= 'models/efficientnet_b3_teacher.pth'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('analysis_results', exist_ok=True)

BATCH_SIZE    = 32
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-3
LR_STEP_SIZE  = 10
GAMMA         = 0.7
ALPHA         = 0.5     # distill loss weight
TEMPERATURE   = 4.0

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# ─── SIMPLE TINYCNN ────────────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

# ─── STUDENT FACTORY ──────────────────────────────────────────────────────────────
def get_student_model(name, ncls):
    if name=='mobilenetv3':
        m = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, ncls)
    elif name=='shufflenet':
        m = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        m.fc = nn.Linear(m.fc.in_features, ncls)
    elif name=='tinycnn':
        m = TinyCNN(ncls)
    else:
        raise ValueError(name)
    return m.to(DEVICE)

# ─── DATALOADERS ────────────────────────────────────────────────────────────────
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

# ─── LOAD MULTI‑TASK TEACHER ─────────────────────────────────────────────────────
def load_teacher(ds, ncls):
    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE)
    # backbone
    backbone = models.efficientnet_b3(weights=None)
    in_feats = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(ckpt['backbone'],strict=False)
    backbone.eval().to(DEVICE)
    # head
    head = nn.Linear(in_feats,ncls).to(DEVICE)
    head.load_state_dict(ckpt['fc_layers'][ds],strict=False)
    head.eval()
    # full teacher = backbone -> head
    return nn.Sequential(backbone, head).eval()

# ─── DISTILLATION LOSS ────────────────────────────────────────────────────────────
def distillation_loss(s_logits,t_logits,targets):
    kd = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(s_logits/TEMPERATURE,1),
        nn.functional.softmax(t_logits/TEMPERATURE,1)
    ) * (TEMPERATURE**2)
    ce = nn.CrossEntropyLoss()(s_logits,targets)
    return ALPHA*kd + (1-ALPHA)*ce

# ─── TRAIN + QAT ─────────────────────────────────────────────────────────────────
def train_qat(student, teacher, tr_ld, val_ld, ds, name):
    student.train()
    student.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    prepare_qat(student, inplace=True)

    opt = torch.optim.Adam(student.parameters(),lr=LEARNING_RATE)
    sch = StepLR(opt,step_size=LR_STEP_SIZE,gamma=GAMMA)

    for ep in range(NUM_EPOCHS):
        tot_loss=0; corr=0; tot=0
        for x,y in tqdm(tr_ld,desc=f"[{name.upper()}][{ds}] Ep{ep+1}/{NUM_EPOCHS}"):
            x,y = x.to(DEVICE), y.squeeze().long().to(DEVICE)
            with torch.no_grad():
                tlog = teacher(x)
            slog = student(x)
            loss = distillation_loss(slog,tlog,y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
            preds = slog.argmax(1)
            corr += (preds==y).sum().item(); tot += y.size(0)

        acc = 100*corr/tot
        print(f" → Loss {tot_loss:.4f} | Acc {acc:.2f}%")
        wandb.log({f"{ds}/{name}/loss":tot_loss,f"{ds}/{name}/acc":acc,"epoch":ep+1})
        sch.step()

    # convert & save
    student.cpu()
    convert(student.eval(),inplace=True)
    out = os.path.join(SAVE_DIR, f"{name}_{ds}_qat.pth")
    torch.save(student.state_dict(), out)
    print(f"💾 Quantized student saved: {out}")

# ─── CPU‑ONLY EVALUATION ──────────────────────────────────────────────────────────
def evaluate(student,val_ld,ds,name):
    student.eval().cpu()
    correct=0; total=0
    with torch.no_grad():
        for x,y in val_ld:
            x,y = x.cpu(), y.squeeze().long().cpu()
            out = student(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item(); total+=y.size(0)

    acc = 100*correct/total
    print(f"🧪 [{name.upper()}][{ds}] QAT Val Acc: {acc:.2f}%")
    wandb.log({f"{ds}/{name}/qat_val_acc":acc})
    with open(f"analysis_results/{name}_{ds}_qat_eval.txt","w") as f:
        f.write(f"{name} on {ds} QAT Acc: {acc:.2f}%\n")

# ─── MAIN ───────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    for student_name in ['mobilenetv3','shufflenet','tinycnn']:
      for ds in DATASETS:
        print(f"\n🔥 QAT+Distill {student_name} on {ds}")
        wandb.init(
          project="medmnist-qat-distill",
          name=f"{student_name}_{ds}_qat",
          config={"epochs":NUM_EPOCHS,"lr":LEARNING_RATE,"batch":BATCH_SIZE,"T":TEMPERATURE,"α":ALPHA}
        )

        tr_ld,val_ld,ncls = get_loader(ds)
        student = get_student_model(student_name,ncls)
        teacher = load_teacher(ds,ncls)

        train_qat(student, teacher, tr_ld, val_ld, ds, student_name)
        evaluate(student, val_ld, ds, student_name)

        wandb.finish()
