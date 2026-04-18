# MH-2026-008 Multiview Boundary Detection — YOLOv8m (Kaggle GPU 版)
# 11-class boundary/valve detection
# 資料集：/kaggle/input/aicad-multiview-boundary-v2/（含 augmented 資料）
# 目標：mAP50 >= 0.70（baseline >= 0.60）
# 預估：< 3 hr on P100
# Sprint 15 P1-1

import os
import sys
import json
import shutil
import random
import subprocess
import time
from pathlib import Path

# torch 2.10+cu128 只含 sm_80+ kernel，T4(sm_75) 會報 cudaErrorNoKernelImageForDevice
# 改裝 cu118（支援 sm_60+），--no-deps 避免動到 numpy
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "t",
    "--index-url", "https://download.pytorch.org/whl/cu121",
    "--no-deps",
])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_using_gpu = DEVICE == "cuda"
print(f"[DEVICE] {DEVICE} (torch={torch.__version__})", flush=True)

REQ_NO = "MH-2026-008"
CLASS_NAMES = [
    "ball_valve", "check_valve", "cock_valve", "flow_meter", "gate_valve",
    "globe_valve", "misc_instrument", "panel", "regulator", "safety_valve",
    "solenoid_valve",
]
NUM_CLASSES = len(CLASS_NAMES)

# Kaggle 資料集掛載路徑
# Kaggle datasets 掛載於 /kaggle/input/datasets/<owner>/<slug>/
# 若不存在則 fallback 到 /kaggle/input/<slug>/（舊版路徑）
_KAGGLE_DATA_CANDIDATES = [
    Path("/kaggle/input/datasets/boardgamegroup/aicad-multiview-boundary-v2"),
    Path("/kaggle/input/aicad-multiview-boundary-v2"),
]
DATA_ROOT = next((p for p in _KAGGLE_DATA_CANDIDATES if p.exists()), _KAGGLE_DATA_CANDIDATES[0])
WORK_DIR = Path("/kaggle/working")
OUT_DIR = WORK_DIR / REQ_NO.lower()
RESULT_PATH = WORK_DIR / "result.json"

YOLO_DIR = WORK_DIR / "dataset"
SEED = 42
EPOCHS = int(os.getenv("MH008_EPOCHS", "100"))
IMGSZ = int(os.getenv("MH008_IMGSZ", "640"))
PATIENCE = 20

print(f"[INIT] req={REQ_NO} epochs={EPOCHS} imgsz={IMGSZ} patience={PATIENCE}", flush=True)
print(f"[INIT] arch=yolov8m classes={NUM_CLASSES}", flush=True)
print(f"[DATA_ROOT] {DATA_ROOT} (exists={DATA_ROOT.exists()})", flush=True)

# Debug：列出 /kaggle/input/ 下所有子目錄，幫助診斷實際掛載路徑
import os as _os
_input_root = Path("/kaggle/input")
if _input_root.exists():
    _found = list(_input_root.rglob("*"))[:30]
    print(f"[DEBUG] /kaggle/input/ entries: {[str(p) for p in _found]}", flush=True)

# ---------------------------------------------------------------------------
# 從 /kaggle/input/aicad-multiview-boundary-v2/ 複製到 working
# yolo_v2_dataset 結構已含 train/ val/ + dataset.yaml
# ---------------------------------------------------------------------------
src_train_images = DATA_ROOT / "train" / "images"
src_train_labels = DATA_ROOT / "train" / "labels"
src_val_images   = DATA_ROOT / "val"   / "images"
src_val_labels   = DATA_ROOT / "val"   / "labels"

for split in ["train", "val"]:
    (YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

def copy_split(src_img_dir, src_lbl_dir, dst_split):
    if not src_img_dir.exists():
        print(f"[WARN] {src_img_dir} does not exist, skipping", flush=True)
        return 0
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = sorted([f for f in src_img_dir.iterdir() if f.suffix.lower() in img_exts])
    count = 0
    for img in imgs:
        dst_img = YOLO_DIR / dst_split / "images" / img.name
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        lbl = src_lbl_dir / (img.stem + ".txt")
        dst_lbl = YOLO_DIR / dst_split / "labels" / lbl.name
        if lbl.exists() and not dst_lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        count += 1
    return count

n_train = copy_split(src_train_images, src_train_labels, "train")
n_val   = copy_split(src_val_images,   src_val_labels,   "val")
print(f"[SPLIT] train={n_train} val={n_val}", flush=True)

# Fallback：若 dataset 為扁平結構（無 train/val 分割）
if n_train == 0:
    print("[FALLBACK] 嘗試從 raw/ + labels_auto/ 做 train/val split", flush=True)
    flat_images = DATA_ROOT / "raw"
    flat_labels = DATA_ROOT / "labels_auto"
    if not flat_images.exists():
        flat_images = DATA_ROOT
        flat_labels = DATA_ROOT
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_imgs = sorted([f for f in flat_images.iterdir() if f.suffix.lower() in img_exts])
    pairs = [(img, flat_labels / (img.stem + ".txt")) for img in all_imgs
             if (flat_labels / (img.stem + ".txt")).exists()]
    print(f"[FALLBACK] found {len(pairs)} pairs", flush=True)
    random.seed(SEED)
    random.shuffle(pairs)
    n_val_split = max(1, int(len(pairs) * 0.2))
    val_pairs   = pairs[:n_val_split]
    train_pairs = pairs[n_val_split:]
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img, label in split_pairs:
            dst_img = YOLO_DIR / split / "images" / img.name
            dst_lbl = YOLO_DIR / split / "labels" / label.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            if label.exists() and not dst_lbl.exists():
                shutil.copy2(label, dst_lbl)
    n_train = len(train_pairs)
    n_val   = len(val_pairs)
    print(f"[FALLBACK SPLIT] train={n_train} val={n_val}", flush=True)

yaml_path = YOLO_DIR / "dataset.yaml"
yaml_path.write_text(
    f"path: {YOLO_DIR}\n"
    f"train: train/images\n"
    f"val: val/images\n\n"
    f"nc: {NUM_CLASSES}\n"
    f"names: {CLASS_NAMES}\n"
)
print(f"[YAML] {yaml_path}", flush=True)

# ---------------------------------------------------------------------------
# Train（yolov8m.pt，從頭 fine-tune）
# ---------------------------------------------------------------------------
from ultralytics import YOLO

print("[MODEL] 載入 yolov8m.pt", flush=True)
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
    mosaic=1.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] train elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

# ---------------------------------------------------------------------------
# Eval + per-class metrics
# ---------------------------------------------------------------------------
best_pt = OUT_DIR / "yolo_run" / "weights" / "best.pt"
if not best_pt.exists():
    best_pt = OUT_DIR / "yolo_run" / "weights" / "last.pt"

model_best = YOLO(str(best_pt))
val_metrics = model_best.val(data=str(yaml_path), imgsz=IMGSZ, device=DEVICE)
map50     = float(val_metrics.box.map50)
map50_95  = float(val_metrics.box.map)
print(f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}", flush=True)

per_class_map50 = {}
try:
    ap_class_index = val_metrics.box.ap_class_index
    ap_per_cls     = val_metrics.box.ap50
    for idx, ap_val in zip(ap_class_index, ap_per_cls):
        name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else str(idx)
        per_class_map50[name] = round(float(ap_val), 4)
    print(f"[PER-CLASS mAP50] {per_class_map50}", flush=True)
except Exception as _pce:
    print(f"[WARN] per-class metrics 提取失敗: {_pce}", flush=True)

if map50 >= 0.88:
    verdict, tier = "pass", "達目標 mAP50>=0.88"
elif map50 >= 0.70:
    verdict, tier = "baseline", "達 baseline mAP50>=0.70"
else:
    verdict, tier = "fail", "未達 baseline"

result = {
    "req_no": REQ_NO,
    "run": "kaggle_v2_cuda",
    "arch": "yolov8s",
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
    "target": "mAP50 >= 0.88",
    "baseline": "mAP50 >= 0.70",
    "classes": CLASS_NAMES,
    "best_path": str(best_pt),
    "augmentation": {"mosaic": 1.0, "hsv_h": 0.015},
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)
