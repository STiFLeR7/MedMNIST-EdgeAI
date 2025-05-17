# scripts/student_qat_run_eval.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from medmnist import INFO
import medmnist
from tqdm import tqdm

# ─── ensure quant backend ─────────────────────────────────────────────────────────
torch.backends.quantized.engine = 'fbgemm'

# ─── CONFIG ───────────────────────────────────────────────────────────────────────
DATASETS      = ['pathmnist', 'dermamnist']
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STUDENT_DIR   = 'models/student_models'
TEACHER_CKPTS = {
    'pathmnist': 'models/efficientnet_b3_teacher_pathmnist.pth',
    'dermamnist': 'models/efficientnet_b3_teacher_dermamnist.pth'
}
BATCH_SIZE    = 32
NUM_QAT_EPOCHS= 5    # fewer QAT epochs
NUM_FT_EPOCHS = 3    # finetune after quantize
LR            = 1e-4

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# ─── STUDENT FACTORY ────────────────────────────────────────────────────────────────
def get_student_model(name, ncls):
    if name=='mobilenetv3':
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, ncls)
    elif name=='shufflenet':
        m = models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Linear(m.fc.in_features, ncls)
    elif name=='tinycnn':
        class TinyCNN(nn.Module):
            def __init__(self, c): 
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
                )
                self.fc = nn.Linear(128, c)
            def forward(self,x):
                x = self.conv(x).view(x.size(0),-1)
                return self.fc(x)
        m = TinyCNN(ncls)
    else:
        raise ValueError(name)
    return m.to(DEVICE)

# ─── DATA LOADER ───────────────────────────────────────────────────────────────────
def get_loader(ds):
    info = INFO[ds]
    cls = getattr(medmnist, info['python_class'])
    tr = cls(split='train', root='data', transform=transform, download=True)
    vl = cls(split='val',   root='data', transform=transform, download=True)
    return (DataLoader(tr, BATCH_SIZE, True),
            DataLoader(vl, BATCH_SIZE, False),
            len(info['label']))

# ─── LOAD TEACHER ─────────────────────────────────────────────────────────────────
def load_teacher(ds, ncls):
    ck = torch.load(TEACHER_CKPTS[ds], map_location=DEVICE)
    t = models.efficientnet_b3(weights=None)
    t.classifier[1] = nn.Linear(t.classifier[1].in_features, ncls)
    t.load_state_dict(ck['backbone'], strict=False)
    return t.eval().to(DEVICE)

# ─── DISTILL LOSS ──────────────────────────────────────────────────────────────────
def distill_loss(s_log, t_log, y, T=4., α=0.5):
    kd = F.kl_div(
        F.log_softmax(s_log/T,1),
        F.softmax(t_log/T,1),
        reduction='batchmean'
    ) * (T*T)
    ce = F.cross_entropy(s_log, y)
    return α*kd + (1-α)*ce

# ─── QAT + DISTILL TRAIN ───────────────────────────────────────────────────────────
def run_qat(student, teacher, tr_ld, ds_name):
    # load pretrained student fp32 weights
    fp32_path = os.path.join(STUDENT_DIR, f"{student_name}_{ds_name}.pth")
    student.load_state_dict(torch.load(fp32_path, map_location=DEVICE), strict=False)

    # prepare QAT
    student.train()
    student.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(student, inplace=True)

    opt = torch.optim.Adam(student.parameters(), lr=LR)

    # QAT epochs
    for ep in range(NUM_QAT_EPOCHS):
        running, corr, tot = 0.,0,0
        for x,y in tqdm(tr_ld, desc=f"[{student_name}][{ds_name}] QAT Ep{ep+1}"):
            x,y = x.to(DEVICE), y.squeeze().long().to(DEVICE)
            with torch.no_grad(): t_log = teacher(x)
            s_log = student(x)
            loss = distill_loss(s_log, t_log, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
            corr   += (s_log.argmax(1)==y).sum().item(); tot += y.size(0)
        print(f" → Ep{ep+1} Loss {running:.3f} Acc {100*corr/tot:.2f}%")

    # convert to int8
    student.cpu()
    torch.quantization.convert(student.eval(), inplace=True)

    # small finetune on CPU int8
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=LR/10)
    for ep in range(NUM_FT_EPOCHS):
        corr, tot = 0,0
        for x,y in tr_ld:
            x,y = x.cpu(), y.squeeze().long().cpu()
            out = student(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            corr += (out.argmax(1)==y).sum().item(); tot += y.size(0)
        print(f"  FT Ep{ep+1} Acc {100*corr/tot:.2f}%")

    save_path = os.path.join(STUDENT_DIR, f"{student_name}_{ds_name}_qat.pth")
    torch.save(student.state_dict(), save_path)
    print("✅ Saved QAT:", save_path)
    return student

# ─── EVAL ───────────────────────────────────────────────────────────────────────────
def evaluate(model, vl, ds_name):
    model.eval()
    corr, tot = 0,0
    with torch.no_grad():
        for x,y in vl:
            x,y = x.cpu(), y.squeeze().long().cpu()
            corr += (model(x).argmax(1)==y).sum().item(); tot += y.size(0)
    acc = 100*corr/tot
    print(f"🧪 [{student_name}][{ds_name}] QAT Val Acc: {acc:.2f}%")
    with open(f"analysis_results/{student_name}_{ds_name}_qat.txt","w") as f:
        f.write(f"{acc:.2f}%\n")

# ─── MAIN ───────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    os.makedirs('analysis_results', exist_ok=True)
    for student_name in ['mobilenetv3','shufflenet','tinycnn']:
      for ds_name in DATASETS:
        print(f"\n🔥 {student_name} on {ds_name} (QAT+Distill+FT)")
        tr_ld, vl, ncls = get_loader(ds_name)
        student = get_student_model(student_name, ncls)
        teacher = load_teacher(ds_name, ncls)
        student = run_qat(student, teacher, tr_ld, ds_name)
        evaluate(student, vl, ds_name)
