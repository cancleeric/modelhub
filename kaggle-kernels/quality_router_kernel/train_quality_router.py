# MH-2026-011 Quality Router — EfficientNet-B0
# 4-class image classification
# 資料集：/kaggle/input/quality-router-dataset/quality_router/
# 目標：val_accuracy >= 0.90
# 預估：< 2 hr（遠低於 9 小時上限）

import os
import sys
import time
import subprocess

# ---------------------------------------------------------------------------
# 1. 安裝相容 P100 (sm_60) 的 PyTorch + timm + numpy 1.x
# 安裝順序關鍵（同 PID/Instrument kernel 做法）：
# 1. torch cu118（P100 sm_60 相容）
# 2. timm（會拉 numpy 2.x）
# 3. numpy==1.26.4 --force-reinstall（最後強制降回 1.x）
# ---------------------------------------------------------------------------
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

import numpy  # numpy 1.26.4 先載入
import timm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np

# ---------------------------------------------------------------------------
# 2. 解壓資料集（各 tier 是獨立 zip，掛載在 aicad-quality-router-flat）
# ---------------------------------------------------------------------------
import zipfile, glob, shutil

# Kaggle 已自動解壓，tier 子目錄直接在 input_root 下
_input_root = "/kaggle/input/aicad-quality-router-flat"
DATA_DIR = _input_root + "/"
print(f"[DATA] DATA_DIR = {DATA_DIR}")
print(f"[DATA] 子目錄: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else '路徑不存在！'}")

# ---------------------------------------------------------------------------
# 3. 設定
# ---------------------------------------------------------------------------
WORK_DIR   = "/kaggle/working/"
BEST_PATH  = os.path.join(WORK_DIR, "quality_router_best.pth")
LAST_PATH  = os.path.join(WORK_DIR, "last.pth")
RESUME_PT  = os.environ.get("RESUME_PT", "")  # 設定 env var 可 resume

NUM_CLASSES = 4
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-3
IMG_SIZE    = 224
SEED        = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device      : {device}")
print(f"timm version: {timm.__version__}")
print(f"torch version: {torch.__version__}")

# ---------------------------------------------------------------------------
# 3. 資料集
# ---------------------------------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 先用無 transform 的 base 做 split，再分別套 transform
base_ds     = datasets.ImageFolder(DATA_DIR)
class_names = base_ds.classes
print(f"Classes     : {class_names}")
print(f"Total images: {len(base_ds)}")

n_val   = int(len(base_ds) * 0.20)
n_train = len(base_ds) - n_val
rng = torch.Generator().manual_seed(SEED)
train_idx, val_idx = random_split(range(len(base_ds)), [n_train, n_val], generator=rng)

train_ds_full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
val_ds_full   = datasets.ImageFolder(DATA_DIR, transform=val_tf)

train_subset = Subset(train_ds_full, list(train_idx))
val_subset   = Subset(val_ds_full,   list(val_idx))

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")

# ---------------------------------------------------------------------------
# 4. 模型（EfficientNet-B0，pretrained，最後一層改 4 類）
# ---------------------------------------------------------------------------
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = GradScaler()  # AMP 混合精度

start_epoch  = 0
best_val_acc = 0.0

# ---------------------------------------------------------------------------
# 5. Resume 支援（設定 RESUME_PT env var）
# ---------------------------------------------------------------------------
if RESUME_PT and os.path.isfile(RESUME_PT):
    print(f"Resuming from {RESUME_PT}")
    ckpt = torch.load(RESUME_PT, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch  = ckpt["epoch"] + 1
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    print(f"  Resumed: epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")

# ---------------------------------------------------------------------------
# 6. 訓練迴圈
# ---------------------------------------------------------------------------
def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
                with autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


print("\n=== Training Start ===")
for epoch in range(start_epoch, EPOCHS):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   training=False)
    scheduler.step()
    elapsed = time.time() - t0

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
          f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  |  "
          f"{elapsed:.1f}s")

    # 儲存 best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
        }, BEST_PATH)
        print(f"  ** Best saved  val_acc={best_val_acc:.4f} -> {BEST_PATH}")

    # 每 5 epoch 存 last.pth
    if (epoch + 1) % 5 == 0:
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
        }, LAST_PATH)
        print(f"  Checkpoint -> last.pth")

# ---------------------------------------------------------------------------
# 7. 最終評估：Confusion Matrix（文字版）
# ---------------------------------------------------------------------------
print("\n=== Final Evaluation ===")
model.eval()
conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs  = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        for t, p in zip(labels.numpy(), preds):
            conf[t][p] += 1

COL_W = 14
print("\nConfusion Matrix (row=actual, col=predicted):")
header = " " * 20 + "".join(f"{c[:COL_W]:>{COL_W}}" for c in class_names)
print(header)
for i, row in enumerate(conf):
    print(f"{class_names[i][:18]:>18}  " + "".join(f"{v:{COL_W}d}" for v in row))

final_acc = conf.diagonal().sum() / conf.sum()
print(f"\nFinal val_acc : {final_acc:.4f}")
print(f"Best  val_acc : {best_val_acc:.4f}")
print(f"Best model    : {BEST_PATH}")

if best_val_acc >= 0.90:
    print("Target ACHIEVED (best_val_acc >= 0.90)")
else:
    print(f"Target NOT reached — best={best_val_acc:.4f}, need 0.90")
