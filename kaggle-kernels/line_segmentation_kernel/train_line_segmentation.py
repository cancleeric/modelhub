# MH-2026-007 Line Segmentation — UNet-ResNet18 (Kaggle GPU 版)
# 工程圖線型語意分割模型
# 資料集：/kaggle/input/mh-2026-007-line-segmentation-dataset/
#   processed/images/*.png  — 512x512 grayscale
#   processed/masks/*.png   — 512x512 grayscale binary mask
# 目標：mIoU >= 0.75
# 預估：< 2 hr on T4/P100

import os
import sys
import json
import random
import subprocess
import time
from pathlib import Path

# ── 安裝相依套件 ────────────────────────────────────────────────────────────────
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.2.2+cu118", "torchvision==0.17.2+cu118",
    "--index-url", "https://download.pytorch.org/whl/cu118",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "segmentation-models-pytorch==0.3.4",
    "albumentations==1.4.3",
    "numpy==1.26.4", "--force-reinstall",
])

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import segmentation_models_pytorch as smp

# ── 常數 ────────────────────────────────────────────────────────────────────────
REQ_NO     = "MH-2026-007"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
WORK_DIR   = Path("/kaggle/working")
BEST_PATH  = WORK_DIR / "line_seg_best.pth"
LAST_PATH  = WORK_DIR / "last.pth"
RESULT_PATH = WORK_DIR / "result.json"

DATA_ROOT  = Path("/kaggle/input/mh-2026-007-line-segmentation-dataset")
IMG_DIR    = DATA_ROOT / "processed" / "images"
MASK_DIR   = DATA_ROOT / "processed" / "masks"

EPOCHS     = int(os.getenv("MH007_EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("MH007_BATCH", "16"))
LR         = float(os.getenv("MH007_LR", "1e-4"))
IMGSZ      = 512
VAL_RATIO  = 0.2
SEED       = 42
MIOU_TARGET = 0.75

print(f"[INIT] req={REQ_NO}  device={DEVICE}", flush=True)
print(f"[DATA] img_dir={IMG_DIR}  mask_dir={MASK_DIR}", flush=True)

# ── Dataset ──────────────────────────────────────────────────────────────────────
class LineSegDataset(Dataset):
    """PNG image + binary mask 配對 Dataset（grayscale → 3-channel 複製）"""

    def __init__(self, pairs, augment=False):
        self.pairs   = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, mask_path = self.pairs[idx]

        img  = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        # binary mask: threshold 0.5
        mask = (mask >= 0.5).astype(np.float32)

        if self.augment:
            # 隨機水平/垂直翻轉
            if random.random() > 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                img  = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            # 隨機旋轉 90 度
            k = random.randint(0, 3)
            if k > 0:
                img  = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()

        # grayscale → 3-channel（ResNet18 backbone 需要）
        img_tensor  = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)  # [3,H,W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)                  # [1,H,W]

        # ImageNet normalize（因為是灰階複製成 3-ch，mean/std 相同）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, mask_tensor

# ── 建立 image-mask 配對清單 ──────────────────────────────────────────────────
img_exts = {".png", ".jpg", ".jpeg"}
all_images = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in img_exts])
pairs = []
for img_path in all_images:
    mask_path = MASK_DIR / img_path.name
    if mask_path.exists():
        pairs.append((img_path, mask_path))

print(f"[PAIRS] {len(pairs)} image-mask pairs found", flush=True)
assert len(pairs) > 0, "找不到 image-mask 配對，請確認 dataset 路徑"

random.seed(SEED)
random.shuffle(pairs)
n_val   = max(1, int(len(pairs) * VAL_RATIO))
n_train = len(pairs) - n_val
train_pairs = pairs[n_train:]   # 後 80%
val_pairs   = pairs[:n_val]     # 前 20%
print(f"[SPLIT] train={n_train}  val={n_val}", flush=True)

train_ds = LineSegDataset(train_pairs, augment=True)
val_ds   = LineSegDataset(val_pairs,   augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── 模型：UNet-ResNet18 ───────────────────────────────────────────────────────
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,   # raw logits，配合 BCEWithLogitsLoss
)
model = model.to(DEVICE)
print(f"[MODEL] UNet-ResNet18  params={sum(p.numel() for p in model.parameters()):,}", flush=True)

# ── Loss & Optimizer ──────────────────────────────────────────────────────────
criterion  = nn.BCEWithLogitsLoss()
optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler     = GradScaler()

# ── IoU 計算工具 ──────────────────────────────────────────────────────────────
def compute_miou(preds_logits: torch.Tensor, targets: torch.Tensor, threshold=0.5) -> float:
    preds = (torch.sigmoid(preds_logits) >= threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    iou   = (inter + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# ── 訓練迴圈 ─────────────────────────────────────────────────────────────────
best_iou = 0.0
t_start  = time.time()
print(f"\n=== Training Start (epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, device={DEVICE}) ===", flush=True)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # Train
    model.train()
    tr_loss, tr_iou_sum, tr_batches = 0.0, 0.0, 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss     += loss.item() * imgs.size(0)
        tr_iou_sum  += compute_miou(logits.detach(), masks) * imgs.size(0)
        tr_batches  += imgs.size(0)

    # Val
    model.eval()
    vl_loss, vl_iou_sum, vl_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, masks)
            vl_loss    += loss.item() * imgs.size(0)
            vl_iou_sum += compute_miou(logits, masks) * imgs.size(0)
            vl_batches += imgs.size(0)

    scheduler.step()
    tr_loss /= tr_batches
    vl_loss /= vl_batches
    tr_iou  = tr_iou_sum / tr_batches
    vl_iou  = vl_iou_sum / vl_batches
    elapsed = time.time() - t0

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}]  "
        f"train_loss={tr_loss:.4f}  train_iou={tr_iou:.4f}  |  "
        f"val_loss={vl_loss:.4f}  val_iou={vl_iou:.4f}  |  {elapsed:.1f}s",
        flush=True,
    )

    if vl_iou > best_iou:
        best_iou = vl_iou
        torch.save(model.state_dict(), BEST_PATH)
        print(f"** Best saved  val_iou={best_iou:.4f} -> {BEST_PATH}", flush=True)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), LAST_PATH)
        print(f"Checkpoint -> last.pth", flush=True)

train_seconds = int(time.time() - t_start)
print(f"\n[DONE] train elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

# ── Final Evaluation ──────────────────────────────────────────────────────────
print("\n=== Final Evaluation ===", flush=True)
model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
model.eval()

final_iou_sum, final_n = 0.0, 0
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with autocast():
            logits = model(imgs)
        final_iou_sum += compute_miou(logits, masks) * imgs.size(0)
        final_n       += imgs.size(0)

final_miou = final_iou_sum / final_n
print(f"Final val mIoU : {final_miou:.4f}", flush=True)
print(f"Best  val mIoU : {best_iou:.4f}",  flush=True)
print(f"Target         : mIoU >= {MIOU_TARGET}", flush=True)

if best_iou >= MIOU_TARGET:
    verdict = "pass"
    print(f"[RESULT] 目標達成 mIoU={best_iou:.4f} >= {MIOU_TARGET}", flush=True)
elif best_iou >= MIOU_TARGET * 0.9:
    verdict = "baseline"
    print(f"[RESULT] baseline 可交付 mIoU={best_iou:.4f}", flush=True)
else:
    verdict = "fail"
    print(f"[RESULT] 未達標 mIoU={best_iou:.4f} < {MIOU_TARGET}", flush=True)

result = {
    "req_no": REQ_NO,
    "device": DEVICE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "imgsz": IMGSZ,
    "train_seconds": train_seconds,
    "n_train": n_train,
    "n_val": n_val,
    "miou": round(best_iou, 4),
    "final_miou": round(final_miou, 4),
    "verdict": verdict,
    "target": f"mIoU >= {MIOU_TARGET}",
    "best_path": str(BEST_PATH),
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

# ── 回報 ModelHub API ─────────────────────────────────────────────────────────
try:
    # Kaggle kernel 中 modelhub_report 透過 inline 方式實作（無法 import 本地模組）
    import requests
    base = os.environ.get("MODELHUB_API_URL", "")
    key  = os.environ.get("MODELHUB_API_KEY", "")
    if base and key:
        payload = {
            "status": "trained" if verdict in ("pass", "baseline") else "training_failed",
            "metrics": {"miou": round(best_iou, 4), "epochs": EPOCHS, "train_seconds": train_seconds},
            "model_path": str(BEST_PATH),
            "notes": f"UNet-ResNet18  val_miou={best_iou:.4f}  verdict={verdict}",
            "per_class_metrics": None,
        }
        resp = requests.patch(
            f"{base}/api/submissions/{REQ_NO}/training-result",
            json=payload,
            headers={"X-Api-Key": key},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"[modelhub_report] {REQ_NO} -> {payload['status']} (HTTP {resp.status_code})", flush=True)
    else:
        print("[modelhub_report] WARN: MODELHUB_API_URL 或 MODELHUB_API_KEY 未設定，跳過回寫", flush=True)
except Exception as e:
    print(f"[modelhub_report] WARN: 回寫失敗，不中斷: {e}", flush=True)

print("\n=== Done ===", flush=True)
