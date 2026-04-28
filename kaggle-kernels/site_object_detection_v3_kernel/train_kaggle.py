# MH-2026-028 Site Object Detection v3 — YOLOv8l (Kaggle GPU)
# CEO Anderson 授權 2026-04-27
# dagongzai 物件偵測，升級 yolov8s → yolov8l + augmentation full
# Dataset: COCO128 + Roboflow supplemental
# Classes: 16 classes (brick/cement/lumber/furniture/tools/appliances)
# Target: mAP50 >= 0.65（full 16-class）
# Estimated: ~3hr on T4

import os
import sys
import json
import shutil
import random
import subprocess
import time
from pathlib import Path

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
    "--index-url", "https://download.pytorch.org/whl/cu121",
    "--no-deps",
])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "roboflow"])

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE} (torch={torch.__version__})", flush=True)

REQ_NO = "MH-2026-028"

TARGET_CLASS_NAMES = [
    "brick",           # 0
    "cement_bag",      # 1
    "lumber",          # 2
    "furniture_chair", # 3
    "furniture_sofa",  # 4
    "furniture_table", # 5
    "cardboard_box",   # 6
    "trash_bag",       # 7
    "broom",           # 8
    "mop",             # 9
    "bucket",          # 10
    "plate_dish",      # 11
    "oven",            # 12
    "refrigerator",    # 13
    "ladder",          # 14
    "hammer",          # 15
]
NUM_CLASSES = len(TARGET_CLASS_NAMES)

COCO_REMAP = {
    56: 3,   # chair        → furniture_chair
    57: 4,   # couch        → furniture_sofa
    60: 5,   # dining table → furniture_table
    68: 12,  # oven         → oven
    72: 13,  # refrigerator → refrigerator
}

WORK_DIR = Path("/kaggle/working")
OUT_DIR = WORK_DIR / REQ_NO.lower()
RESULT_PATH = WORK_DIR / "result.json"
YOLO_DIR = WORK_DIR / "dataset"
SEED = 42
# v3: yolov8l + 250 epochs（v2 yolov8s 200 epochs）
EPOCHS = int(os.getenv("MH028_EPOCHS", "250"))
IMGSZ = int(os.getenv("MH028_IMGSZ", "640"))
PATIENCE = 80

print(f"[INIT] req={REQ_NO} epochs={EPOCHS} imgsz={IMGSZ} patience={PATIENCE}", flush=True)
print(f"[INIT] arch=yolov8l classes={NUM_CLASSES}", flush=True)
print(f"[DIFF vs v2] model: yolov8s→yolov8l, epochs: 200→250, augmentation: full", flush=True)

_input_root = Path("/kaggle/input")
if _input_root.exists():
    _dirs = [str(p) for p in _input_root.rglob("*") if p.is_dir()][:20]
    print(f"[DEBUG] /kaggle/input/ dirs: {_dirs}", flush=True)

_COCO_CANDIDATES = [
    Path("/kaggle/input/datasets/organizations/ultralytics/coco128/coco128"),
    Path("/kaggle/input/datasets/ultralytics/coco128"),
    Path("/kaggle/input/coco128/coco128"),
    Path("/kaggle/input/coco128"),
]
COCO_ROOT = next((p for p in _COCO_CANDIDATES if p.exists()), _COCO_CANDIDATES[0])
print(f"[COCO_ROOT] {COCO_ROOT} (exists={COCO_ROOT.exists()})", flush=True)

for split in ["train", "val"]:
    (YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def remap_coco_label(src_label: Path, dst_label: Path, remap: dict) -> int:
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
    return 0


# COCO128
coco_img_dir = COCO_ROOT / "images" / "train2017"
coco_lbl_dir = COCO_ROOT / "labels" / "train2017"
if not coco_img_dir.exists():
    found = list(COCO_ROOT.rglob("train2017"))
    if found:
        coco_img_dir = found[0] if "images" in str(found[0]) else found[0]
        coco_lbl_dir = Path(str(coco_img_dir).replace("images", "labels"))
    else:
        coco_img_dir = COCO_ROOT
        coco_lbl_dir = COCO_ROOT

all_coco = sorted([f for f in coco_img_dir.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]) \
    if coco_img_dir.exists() else []
random.seed(SEED)
random.shuffle(all_coco)
n_val_coco = max(1, int(len(all_coco) * 0.2))
coco_val = all_coco[:n_val_coco]
coco_train = all_coco[n_val_coco:]

n_coco_train = 0
for img in coco_train:
    lbl = coco_lbl_dir / (img.stem + ".txt")
    dst_img = YOLO_DIR / "train" / "images" / img.name
    dst_lbl = YOLO_DIR / "train" / "labels" / (img.stem + ".txt")
    lines_kept = remap_coco_label(lbl, dst_lbl, COCO_REMAP)
    if lines_kept > 0:
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        n_coco_train += 1

n_coco_val = 0
for img in coco_val:
    lbl = coco_lbl_dir / (img.stem + ".txt")
    dst_img = YOLO_DIR / "val" / "images" / img.name
    dst_lbl = YOLO_DIR / "val" / "labels" / (img.stem + ".txt")
    lines_kept = remap_coco_label(lbl, dst_lbl, COCO_REMAP)
    if lines_kept > 0:
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        n_coco_val += 1

print(f"[COCO128] mapped train={n_coco_train} val={n_coco_val}", flush=True)

# Roboflow supplemental
RF_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
n_rf_train = 0
if RF_API_KEY:
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=RF_API_KEY)
        project = rf.workspace("roboflow-universe-projects").project("construction-site-safety-g9e2b")
        dataset = project.version(4).download("yolov8", location="/kaggle/working/rf_dataset")
        print("[ROBOFLOW] download success", flush=True)
    except Exception as _rf_err:
        print(f"[ROBOFLOW] failed: {_rf_err}", flush=True)
else:
    print("[ROBOFLOW] ROBOFLOW_API_KEY not set — skipping", flush=True)

n_train_total = len(list((YOLO_DIR / "train" / "images").glob("*")))
n_val_total = len(list((YOLO_DIR / "val" / "images").glob("*")))
print(f"[DATASET] final train={n_train_total} val={n_val_total}", flush=True)

yaml_path = YOLO_DIR / "dataset.yaml"
yaml_path.write_text(
    f"path: {YOLO_DIR}\n"
    f"train: train/images\n"
    f"val: val/images\n\n"
    f"nc: {NUM_CLASSES}\n"
    f"names: {TARGET_CLASS_NAMES}\n"
)

# ---------------------------------------------------------------------------
# Train — yolov8l + augmentation full
# ---------------------------------------------------------------------------
from ultralytics import YOLO

print("[MODEL] Loading yolov8l.pt", flush=True)
model = YOLO("yolov8l.pt")

t_start = time.time()
print(f"\n=== Training Start (yolov8l, epochs={EPOCHS}, patience={PATIENCE}) ===", flush=True)
model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=8,          # yolov8l 更大模型，batch 調小防 OOM
    device=DEVICE,
    project=str(OUT_DIR),
    name="yolo_run",
    exist_ok=True,
    patience=PATIENCE,
    save=True,
    plots=True,
    verbose=True,
    # augmentation full（比 v2 更激進）
    mosaic=1.0,
    mixup=0.2,
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.1,
    scale=0.6,
    degrees=15.0,
    copy_paste=0.2,
    translate=0.1,
    # cosine LR
    cos_lr=True,
    lr0=0.01,
    lrf=0.005,
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

MAP50_TARGET = 0.65
MAP50_BASELINE = 0.35
if map50 >= MAP50_TARGET:
    verdict, tier = "pass", f"達目標 mAP50>={MAP50_TARGET}"
elif map50 >= MAP50_BASELINE:
    verdict, tier = "baseline", f"達 baseline mAP50>={MAP50_BASELINE}"
else:
    verdict, tier = "fail", "未達 baseline（COCO128 只有 5/16 class coverage）"

result = {
    "req_no": REQ_NO,
    "run": "kaggle_v3_yolov8l",
    "arch": "yolov8l",
    "device": DEVICE,
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "patience": PATIENCE,
    "n_train": n_train_total,
    "n_val": n_val_total,
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
    "dataset_coverage": {
        "coco128_classes": ["furniture_chair", "furniture_sofa", "furniture_table", "oven", "refrigerator"],
        "missing_classes": ["brick", "cement_bag", "lumber", "cardboard_box", "trash_bag",
                            "broom", "mop", "bucket", "plate_dish", "ladder", "hammer"],
        "note": "Phase 1 baseline 5/16 COCO-covered classes. yolov8l provides more capacity for captured classes.",
    },
    "note": "v3: yolov8l (vs v2 yolov8s), epochs=250, augmentation full, batch=8",
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)
