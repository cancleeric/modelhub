# MH-2026-018 PPE Detection — YOLOv8m (Kaggle GPU)
# 7-class PPE detection for construction site workers
# Dataset: Construction Site Safety Image Dataset (Roboflow) by snehilsanyal
# Classes: person, hardhat, no_hardhat, safety_vest, no_safety_vest, mask, gloves
# Target: mAP50 >= 0.75
# Estimated: < 4 hr on T4
# Priority: P1
# PII Policy: dataset contains worker images — no face embeddings or biometric data extracted

import os
import sys
import json
import shutil
import random
import subprocess
import time
from pathlib import Path

# torch 2.10+cu128 only supports sm_80+ kernels, T4 is sm_75 which will fail
# Install cu121 (supports sm_60+) with --no-deps to avoid touching numpy
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
    "--index-url", "https://download.pytorch.org/whl/cu121",
    "--no-deps",
])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE} (torch={torch.__version__})", flush=True)

REQ_NO = "MH-2026-018"

# Target 7 classes for MH-2026-018
# The CSS Roboflow dataset has 10 classes. We map what we can and
# treat missing label IDs as background (they simply won't appear in predictions).
TARGET_CLASS_NAMES = [
    "person",
    "hardhat",
    "no_hardhat",
    "safety_vest",
    "no_safety_vest",
    "mask",
    "gloves",
]

# CSS dataset (snehilsanyal/construction-site-safety-image-dataset-roboflow) classes:
# 0=Hardhat 1=Mask 2=NO-Hardhat 3=NO-Safety Vest 4=NO-mask 5=Person
# 6=Safety Cone 7=Safety Vest 8=machinery 9=vehicle
# We remap to our 7-class schema: person(5→0), hardhat(0→1), no_hardhat(2→2),
# safety_vest(7→3), no_safety_vest(3→4), mask(1→5), gloves(none—skip)
# Remapping done at data copy time so YOLO only sees 7 classes.
CSS_REMAP = {
    5: 0,  # Person      → person
    0: 1,  # Hardhat     → hardhat
    2: 2,  # NO-Hardhat  → no_hardhat
    7: 3,  # Safety Vest → safety_vest
    3: 4,  # NO-Safety Vest → no_safety_vest
    1: 5,  # Mask        → mask
    # 6=Safety Cone, 8=machinery, 9=vehicle → discarded
    # gloves class(6 in our schema) → no mapping in CSS dataset, stays 0 instances
}

NUM_CLASSES = len(TARGET_CLASS_NAMES)

# Kaggle dataset mount paths
_KAGGLE_DATA_CANDIDATES = [
    Path("/kaggle/input/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow"),
    Path("/kaggle/input/construction-site-safety-image-dataset-roboflow"),
]
DATA_ROOT = next((p for p in _KAGGLE_DATA_CANDIDATES if p.exists()), _KAGGLE_DATA_CANDIDATES[0])
WORK_DIR = Path("/kaggle/working")
OUT_DIR = WORK_DIR / REQ_NO.lower()
RESULT_PATH = WORK_DIR / "result.json"
YOLO_DIR = WORK_DIR / "dataset"
SEED = 42
EPOCHS = int(os.getenv("MH018_EPOCHS", "150"))  # v2: 100→150
IMGSZ = int(os.getenv("MH018_IMGSZ", "640"))
PATIENCE = 60

print(f"[INIT] req={REQ_NO} epochs={EPOCHS} imgsz={IMGSZ} patience={PATIENCE}", flush=True)
print(f"[INIT] arch=yolov8m classes={NUM_CLASSES}", flush=True)
print(f"[DATA_ROOT] {DATA_ROOT} (exists={DATA_ROOT.exists()})", flush=True)

# Debug: list /kaggle/input/ contents to diagnose mount path
_input_root = Path("/kaggle/input")
if _input_root.exists():
    _found = [str(p) for p in _input_root.rglob("*") if p.is_dir()][:20]
    print(f"[DEBUG] /kaggle/input/ dirs: {_found}", flush=True)


def remap_label_file(src_label: Path, dst_label: Path, remap: dict) -> int:
    """
    Copy YOLO label file, remapping class IDs via remap dict.
    Lines whose class ID is not in remap are discarded.
    Returns count of kept lines.
    """
    if not src_label.exists():
        return 0
    kept = []
    for line in src_label.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        old_cls = int(parts[0])
        if old_cls in remap:
            parts[0] = str(remap[old_cls])
            kept.append(" ".join(parts))
    if kept:
        dst_label.write_text("\n".join(kept) + "\n")
    return len(kept)


def copy_split_remapped(src_img_dir: Path, src_lbl_dir: Path, dst_split: str) -> int:
    """Copy images and remap labels for a given split."""
    if not src_img_dir.exists():
        print(f"[WARN] {src_img_dir} does not exist, skipping", flush=True)
        return 0
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted([f for f in src_img_dir.iterdir() if f.suffix.lower() in img_exts])
    count = 0
    for img in imgs:
        dst_img = YOLO_DIR / dst_split / "images" / img.name
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        lbl = src_lbl_dir / (img.stem + ".txt")
        dst_lbl = YOLO_DIR / dst_split / "labels" / lbl.name
        if not dst_lbl.exists():
            remap_label_file(lbl, dst_lbl, CSS_REMAP)
        count += 1
    return count


for split in ["train", "val"]:
    (YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# CSS dataset structure: css-data/train/images, css-data/train/labels, css-data/valid/...
css_data = DATA_ROOT / "css-data"
if not css_data.exists():
    # Fallback: dataset root directly
    css_data = DATA_ROOT

n_train = copy_split_remapped(
    css_data / "train" / "images", css_data / "train" / "labels", "train"
)
# CSS uses 'valid' for validation split
n_val = copy_split_remapped(
    css_data / "valid" / "images", css_data / "valid" / "labels", "val"
)
if n_val == 0:
    n_val = copy_split_remapped(
        css_data / "val" / "images", css_data / "val" / "labels", "val"
    )

print(f"[SPLIT] train={n_train} val={n_val}", flush=True)

# Fallback: flat directory structure
if n_train == 0:
    print("[FALLBACK] flat directory split", flush=True)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_imgs = sorted([f for f in DATA_ROOT.rglob("*.jpg")] +
                      [f for f in DATA_ROOT.rglob("*.png")])
    all_imgs = [f for f in all_imgs if "images" in str(f)]
    lbl_candidates = {}
    for img in all_imgs:
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        if lbl.exists():
            lbl_candidates[img] = lbl
    items = list(lbl_candidates.items())
    random.seed(SEED)
    random.shuffle(items)
    n_val_split = max(1, int(len(items) * 0.15))
    val_items = items[:n_val_split]
    train_items = items[n_val_split:]
    for split, split_items in [("train", train_items), ("val", val_items)]:
        for img, lbl in split_items:
            dst_img = YOLO_DIR / split / "images" / img.name
            dst_lbl = YOLO_DIR / split / "labels" / (img.stem + ".txt")
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            if not dst_lbl.exists():
                remap_label_file(lbl, dst_lbl, CSS_REMAP)
    n_train = len(train_items)
    n_val = len(val_items)
    print(f"[FALLBACK SPLIT] train={n_train} val={n_val}", flush=True)

yaml_path = YOLO_DIR / "dataset.yaml"
yaml_path.write_text(
    f"path: {YOLO_DIR}\n"
    f"train: train/images\n"
    f"val: val/images\n\n"
    f"nc: {NUM_CLASSES}\n"
    f"names: {TARGET_CLASS_NAMES}\n"
)
print(f"[YAML] {yaml_path}", flush=True)

# ---------------------------------------------------------------------------
# Train (yolov8m.pt, fine-tune from pretrained)
# ---------------------------------------------------------------------------
from ultralytics import YOLO

print("[MODEL] Loading yolov8m.pt", flush=True)
model = YOLO("yolov8m.pt")

t_start = time.time()
print(f"\n=== Training Start (yolov8m, epochs={EPOCHS}, patience={PATIENCE}) ===", flush=True)
model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=16,
    device=DEVICE,
    project=str(OUT_DIR),
    name="yolo_run",
    exist_ok=True,
    patience=PATIENCE,
    save=True,
    plots=True,
    verbose=True,
    # v2 augmentation: mosaic + mixup（YOLOv8 內建）
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.1,
    fliplr=0.5,
    scale=0.5,
    degrees=5.0,
    copy_paste=0.1,
    # v2: cosine learning rate scheduler
    cos_lr=True,
    lr0=0.01,
    lrf=0.01,
)
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] train elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
best_pt = OUT_DIR / "yolo_run" / "weights" / "best.pt"
if not best_pt.exists():
    best_pt = OUT_DIR / "yolo_run" / "weights" / "last.pt"

model_best = YOLO(str(best_pt))
val_metrics = model_best.val(data=str(yaml_path), imgsz=IMGSZ, device=DEVICE)
map50 = float(val_metrics.box.map50)
map50_95 = float(val_metrics.box.map)
print(f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}", flush=True)

per_class_map50 = {}
try:
    ap_class_index = val_metrics.box.ap_class_index
    ap_per_cls = val_metrics.box.ap50
    for idx, ap_val in zip(ap_class_index, ap_per_cls):
        name = TARGET_CLASS_NAMES[int(idx)] if int(idx) < len(TARGET_CLASS_NAMES) else str(idx)
        per_class_map50[name] = round(float(ap_val), 4)
    print(f"[PER-CLASS mAP50] {per_class_map50}", flush=True)
except Exception as _pce:
    print(f"[WARN] per-class metrics extraction failed: {_pce}", flush=True)

MAP50_TARGET = 0.75
MAP50_BASELINE = 0.55
if map50 >= MAP50_TARGET:
    verdict, tier = "pass", f"達目標 mAP50>={MAP50_TARGET}"
elif map50 >= MAP50_BASELINE:
    verdict, tier = "baseline", f"達 baseline mAP50>={MAP50_BASELINE}"
else:
    verdict, tier = "fail", "未達 baseline"

result = {
    "req_no": REQ_NO,
    "run": "kaggle_v2_cuda",
    "arch": "yolov8m",
    "device": DEVICE,
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "patience": PATIENCE,
    "n_train": n_train,
    "n_val": n_val,
    "train_seconds": train_seconds,
    "map50": round(map50, 4),
    "map50_95": round(map50_95, 4),
    "per_class_map50": per_class_map50,
    "verdict": verdict,
    "tier": tier,
    "target": f"mAP50 >= {MAP50_TARGET}",
    "baseline": f"mAP50 >= {MAP50_BASELINE}",
    "classes": TARGET_CLASS_NAMES,
    "best_path": str(best_pt),
    "dataset": "snehilsanyal/construction-site-safety-image-dataset-roboflow",
    "note": "CSS dataset class remapped to 7-class PPE schema. Gloves class has 0 training samples from CSS dataset.",
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)
