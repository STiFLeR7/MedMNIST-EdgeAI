# utils/fine_tuning_qt.py

import os, glob, torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from medmnist import INFO, __dict__ as medmnist_dict
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.quantization import (
    QConfig, prepare_qat,
    default_fake_quant, default_weight_fake_quant
)

# --- Force the backend for dynamic quant ---
torch.backends.quantized.engine = 'fbgemm'

# --- CONFIG ---
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT   = 'data'
MODEL_DIR   = 'models'
QUANT_DIR   = 'quantization/models'
BATCH_SIZE  = 128
Q_EPOCHS    = 5
LR          = 1e-4
T           = 3.0
ALPHA       = 0.5
DOWNLOAD    = False
SEED        = 42
torch.manual_seed(SEED)

# --- Distillation loss ---
def distill_loss(s, t, y):
    kd = F.kl_div(
        F.log_softmax(s/T,1), F.softmax(t/T,1),
        reduction='batchmean'
    ) * (T*T)
    ce = F.cross_entropy(s, y)
    return ALPHA*kd + (1-ALPHA)*ce

# --- QAT config: fake‐quant for activations & weights (per‐tensor) ---
qat_qconfig = QConfig(
    activation=default_fake_quant,
    weight=default_weight_fake_quant
)

# --- Student factory (same as analysis.py) ---
def get_student_model(mt, in_ch, n_cls):
    if mt=='mobilenetv3':
        m = torch.hub.load('pytorch/vision','mobilenet_v3_small',pretrained=False)
        m.features[0][0] = nn.Conv2d(in_ch,16,3,2,1,bias=False)
        m.classifier[3]  = nn.Linear(m.classifier[3].in_features,n_cls)
        return m
    if mt=='shufflenetv2':
        m = torch.hub.load('pytorch/vision','shufflenet_v2_x1_0',pretrained=False)
        m.conv1[0] = nn.Conv2d(in_ch,24,3,2,1,bias=False)
        m.fc       = nn.Linear(m.fc.in_features,n_cls)
        return m
    # tinycnn
    layers = [
        nn.Conv2d(in_ch,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,1,1),   nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,1,1),  nn.ReLU(), nn.MaxPool2d(2),
    ]
    feat = nn.Sequential(*layers)
    with torch.no_grad():
        flat = feat(torch.zeros(1,in_ch,28,28)).view(1,-1).size(1)
    return nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(flat, n_cls)
    )

# --- Teacher loader (ResNet-18) ---
def load_teacher(in_ch,n_cls):
    t = torch.hub.load('pytorch/vision','resnet18',pretrained=False)
    t.conv1 = nn.Conv2d(in_ch,64,7,2,3,bias=False)
    t.fc    = nn.Linear(t.fc.in_features,n_cls)
    ck  = os.path.join(MODEL_DIR,'resnet18_mnist.pth')
    t.load_state_dict(torch.load(ck, map_location=DEVICE), strict=False)
    return t.to(DEVICE).eval()

# --- QAT fine‐tune ---
def fine_tune_qat(model, teacher, loader):
    model.train()
    model.qconfig = qat_qconfig
    prepare_qat(model, inplace=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(Q_EPOCHS):
        tot, corr, loss_sum = 0,0,0.
        pbar = tqdm(loader, desc=f"QAT Ep {ep+1}/{Q_EPOCHS}")
        for x,y in pbar:
            x,y = x.to(DEVICE), y.squeeze().long().to(DEVICE)
            with torch.no_grad(): tgt = teacher(x)
            out = model(x)
            loss = distill_loss(out, tgt, y)
            opt.zero_grad(); loss.backward(); opt.step()

            loss_sum += loss.item()
            preds = out.argmax(1)
            corr    += (preds==y).sum().item()
            tot     += y.size(0)
        pbar.write(f" → Loss {loss_sum:.4f} | Acc {100*corr/tot:.2f}%")

    model.eval()
    # --- instead of static convert, use dynamic quant on the Linears only ---
    model.cpu()
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized

# --- Main loop ---
def main():
    os.makedirs(QUANT_DIR, exist_ok=True)
    for ckpt in glob.glob(os.path.join(QUANT_DIR,'*_quantized.pth')):
        name = os.path.basename(ckpt).rsplit('.',1)[0]
        mt, ds,_ = name.split('_',2)
        print(f"\n⏳ Fine-tuning QAT for {mt} on {ds}")

        info        = INFO[ds]
        in_ch,n_cls = info['n_channels'], len(info['label'])
        teacher     = load_teacher(in_ch,n_cls)

        # build train loader
        Dclass = medmnist_dict[info['python_class']]
        tx = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([.5]*in_ch,[.5]*in_ch)])
        tr = Dclass(split='train',root=DATA_ROOT,transform=tx,download=DOWNLOAD)
        ld = DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True)

        # load float student
        float_ck = os.path.join(MODEL_DIR,f"{mt}_{ds}_student.pth")
        student  = get_student_model(mt,in_ch,n_cls).to(DEVICE)
        student.load_state_dict(torch.load(float_ck,map_location=DEVICE),strict=False)

        # fine-tune + dynamic quantize
        qmodel = fine_tune_qat(student, teacher, ld)

        # overwrite the quantized checkpoint
        torch.save(qmodel.state_dict(), ckpt)
        print(f"✅ Overwrote QAT quantized file: {ckpt}")

if __name__=='__main__':
    main()
