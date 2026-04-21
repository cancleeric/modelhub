#!/usr/bin/env python3
# MH-2026-007 Line Segmentation — Local Eval Script
# 使用 Kaggle 訓練好的 weights（mh_2026_007_line_seg_best.pth）在本機 val set 上跑 eval
# device: mps（本機 Apple Silicon），eval-only，不訓練

import os
import sys
import json
import random
import time
from pathlib import Path

# ── 相依套件確認 ──────────────────────────────────────────────────────────────
try:
    import torch
    import segmentation_models_pytorch as smp
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"[ERROR] 缺少套件：{e}")
    print("[HINT] pip install torch torchvision segmentation-models-pytorch Pillow numpy")
    sys.exit(1)

# ── 常數 ──────────────────────────────────────────────────────────────────────
REQ_NO     = "MH-2026-007"
# 本機 MPS（eval only，不訓練，規範允許）
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
WEIGHTS    = Path(__file__).parent.parent.parent / "weights" / "mh_2026_007_line_seg_best.pth"
DATA_ROOT  = Path(os.environ.get(
    "LINE_SEG_DATASET",
    "/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/line_segmentation/processed"
))
IMG_DIR    = DATA_ROOT / "images"
MASK_DIR   = DATA_ROOT / "masks"
RESULT_PATH = Path(__file__).parent / "eval_result.json"
IMGSZ      = 512
VAL_RATIO  = 0.2
SEED       = 42
MIOU_TARGET = 0.75

print(f"[INIT] req={REQ_NO}  device={DEVICE}", flush=True)
print(f"[WEIGHTS] {WEIGHTS}  exists={WEIGHTS.exists()}", flush=True)
print(f"[DATA] {DATA_ROOT}  exists={DATA_ROOT.exists()}", flush=True)

assert WEIGHTS.exists(), f"Weights 不存在：{WEIGHTS}"
assert IMG_DIR.exists(),  f"Images 目錄不存在：{IMG_DIR}"
assert MASK_DIR.exists(), f"Masks 目錄不存在：{MASK_DIR}"

# ── Dataset ───────────────────────────────────────────────────────────────────
img_exts = {".png", ".jpg", ".jpeg"}
all_images = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in img_exts])
pairs = []
for img_path in all_images:
    mask_path = MASK_DIR / img_path.name
    if mask_path.exists():
        pairs.append((img_path, mask_path))

print(f"[PAIRS] {len(pairs)} image-mask pairs found", flush=True)
assert len(pairs) > 0, "找不到 image-mask 配對"

random.seed(SEED)
random.shuffle(pairs)
n_val = max(1, int(len(pairs) * VAL_RATIO))
# 與訓練時一致：前 20% 為 val
val_pairs = pairs[:n_val]
print(f"[VAL] {n_val} pairs for evaluation", flush=True)

def load_pair(img_path, mask_path):
    img  = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    mask = (mask >= 0.5).astype(np.float32)
    img_tensor  = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)  # [3,H,W]
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)                  # [1,H,W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor, mask_tensor

def compute_miou_batch(preds_logits, targets, threshold=0.5):
    preds = (torch.sigmoid(preds_logits) >= threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    iou   = (inter + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# ── 載入模型 ──────────────────────────────────────────────────────────────────
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,   # eval 時不需要 imagenet pretrain
    in_channels=3,
    classes=1,
    activation=None,
)
state_dict = torch.load(str(WEIGHTS), map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()
print(f"[MODEL] UNet-ResNet18 loaded from {WEIGHTS.name}", flush=True)

# ── Evaluation ────────────────────────────────────────────────────────────────
BATCH_SIZE = 8
iou_sum = 0.0
n_total = 0
t_start = time.time()

print(f"\n=== Evaluation Start (n={n_val}, batch={BATCH_SIZE}, device={DEVICE}) ===", flush=True)

with torch.no_grad():
    for i in range(0, len(val_pairs), BATCH_SIZE):
        batch = val_pairs[i:i + BATCH_SIZE]
        imgs_list, masks_list = [], []
        for img_path, mask_path in batch:
            img_t, mask_t = load_pair(img_path, mask_path)
            imgs_list.append(img_t)
            masks_list.append(mask_t)
        imgs  = torch.stack(imgs_list).to(DEVICE)
        masks = torch.stack(masks_list).to(DEVICE)
        logits = model(imgs)
        batch_iou = compute_miou_batch(logits, masks)
        iou_sum += batch_iou * len(batch)
        n_total += len(batch)
        print(f"  batch [{i//BATCH_SIZE + 1}] iou={batch_iou:.4f}  ({n_total}/{n_val})", flush=True)

final_miou = iou_sum / n_total
elapsed    = int(time.time() - t_start)

print(f"\n[RESULT] mIoU = {final_miou:.4f}", flush=True)
print(f"[TARGET] mIoU >= {MIOU_TARGET}", flush=True)
print(f"[ELAPSED] {elapsed}s", flush=True)

if final_miou >= MIOU_TARGET:
    verdict = "pass"
    print(f"[VERDICT] 目標達成 mIoU={final_miou:.4f} >= {MIOU_TARGET}", flush=True)
elif final_miou >= MIOU_TARGET * 0.9:
    verdict = "baseline"
    print(f"[VERDICT] baseline 可交付 mIoU={final_miou:.4f}", flush=True)
else:
    verdict = "fail"
    print(f"[VERDICT] 未達標 mIoU={final_miou:.4f} < {MIOU_TARGET}", flush=True)

result = {
    "req_no": REQ_NO,
    "method": "local_eval_kaggle_weights",
    "device": DEVICE,
    "weights": str(WEIGHTS),
    "n_val": n_val,
    "miou": round(final_miou, 4),
    "target": f"mIoU >= {MIOU_TARGET}",
    "verdict": verdict,
    "eval_seconds": elapsed,
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
print("\n=== Done ===", flush=True)
