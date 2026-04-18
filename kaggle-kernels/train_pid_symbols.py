# MH-2026-010 PID Symbols — MobileNetV2
# 11-class image classification
# 資料集：/kaggle/input/aicad-pid-symbols-flat/ (11 class subdirs)
# 目標：val_accuracy >= 0.90（misc_instrument >= 0.85）
# 預估：< 1 hr

import os, sys, time, subprocess

# 安裝順序：torch cu118 → timm → numpy 1.26.4（同 PID kernel）
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.2.2+cu118", "torchvision==0.17.2+cu118",
    "--index-url", "https://download.pytorch.org/whl/cu118",
])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm"])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy==1.26.4", "--force-reinstall",
])

import numpy
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR  = "/kaggle/input/aicad-pid-symbols-flat/"
WORK_DIR  = "/kaggle/working/"
BEST_PATH = os.path.join(WORK_DIR, "pid_symbols_best.pth")
LAST_PATH = os.path.join(WORK_DIR, "last.pth")
RESUME_PT = os.environ.get("RESUME_PT", "")

NUM_CLASSES = 11
EPOCHS      = 40
BATCH_SIZE  = 64
LR          = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device      : {device}")
print(f"timm version: {timm.__version__}")
print(f"torch version: {torch.__version__}")
print(f"DATA_DIR    : {DATA_DIR}")
print(f"Classes     : {sorted(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else 'N/A'}")

train_tf = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
val_tf = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

full_ds = datasets.ImageFolder(DATA_DIR)
classes = full_ds.classes
print(f"Total images: {len(full_ds)}")
n_val = int(len(full_ds) * 0.2)
n_train = len(full_ds) - n_val
train_idx, val_idx = random_split(range(len(full_ds)), [n_train, n_val],
                                   generator=torch.Generator().manual_seed(42))

train_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(DATA_DIR, transform=val_tf)
from torch.utils.data import Subset
train_ds = Subset(train_ds, train_idx.indices)
val_ds   = Subset(val_ds,   val_idx.indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

model = timm.create_model("mobilenetv2_100", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

if RESUME_PT and os.path.exists(RESUME_PT):
    model.load_state_dict(torch.load(RESUME_PT, map_location=device))
    print(f"[RESUME] 從 {RESUME_PT} 繼續")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler()

best_acc = 0.0

def run_epoch(loader, training=True):
    model.train(training)
    total_loss, correct, total = 0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

print("\n=== Training Start ===")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    vl_loss, vl_acc = run_epoch(val_loader,   training=False)
    scheduler.step()
    elapsed = time.time() - t0
    print(f"Epoch [{epoch:02d}/{EPOCHS}]  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  |  val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  |  {elapsed:.1f}s")
    if vl_acc > best_acc:
        best_acc = vl_acc
        torch.save(model.state_dict(), BEST_PATH)
        print(f"** Best saved  val_acc={best_acc:.4f} -> {BEST_PATH}")
    if epoch % 5 == 0:
        torch.save(model.state_dict(), LAST_PATH)
        print(f"Checkpoint -> last.pth")

# Final evaluation with per-class accuracy
print("\n=== Final Evaluation ===")
model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.eval()
class_correct = [0] * NUM_CLASSES
class_total   = [0] * NUM_CLASSES
conf_matrix   = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for t, p in zip(labels, preds):
            conf_matrix[t.item()][p.item()] += 1
            class_total[t.item()] += 1
            if t == p:
                class_correct[t.item()] += 1

print("Per-class accuracy:")
misc_acc = 0.0
for i, cls in enumerate(classes):
    acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    flag = " ← misc" if cls == "misc_instrument" else ""
    if cls == "misc_instrument":
        misc_acc = acc
    print(f"  {cls:30s}: {acc:.4f}{flag}")

final_acc = sum(class_correct) / sum(class_total)
print(f"\nFinal val_acc : {final_acc:.4f}")
print(f"Best  val_acc : {best_acc:.4f}")
print(f"misc_instrument acc: {misc_acc:.4f} (target >= 0.85)")

if best_acc >= 0.90 and misc_acc >= 0.85:
    print("✅ 目標達成 (val_acc >= 0.90, misc >= 0.85)")
elif best_acc >= 0.90:
    print(f"⚠️  val_acc 達標但 misc_instrument ({misc_acc:.4f}) < 0.85")
else:
    print(f"❌ val_acc ({best_acc:.4f}) < 0.90，需繼續訓練")
