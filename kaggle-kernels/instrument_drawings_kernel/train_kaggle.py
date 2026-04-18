# MH-2026-005 Instrument Drawings — YOLOv8m fine-tune (Kaggle GPU v2)
# 11-class PID symbol detection
# 資料集：/kaggle/input/aicad-instrument-drawings/
# 目標：mAP50 >= 0.92（target），>= 0.70（baseline）
# 架構：yolov8m（由 yolov8s 升級，大幅提升表達能力）
# 預估：< 4 hr on T4

import os
import sys
import json
import shutil
import random
import subprocess
import time
from pathlib import Path

# torch cu128 只含 sm_80+，T4(sm_75) 報 cudaErrorNoKernelImageForDevice
# torch 2.2.x + Python3.12 numpy2.x 報 Numpy is not available
# 解法：cu121 含 sm_75 kernel，torch>=2.4 支援 numpy 2.x
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
    "--index-url", "https://download.pytorch.org/whl/cu121",
    "--no-deps",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "ultralytics",
])

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE} (torch={torch.__version__})", flush=True)

REQ_NO = "MH-2026-005"
# Kaggle 資料集掛載路徑（zip 上傳後會解壓到此）
_KAGGLE_DATA_CANDIDATES = [
    Path("/kaggle/input/aicad-instrument-drawings"),
    Path("/kaggle/input/datasets/boardgamegroup/aicad-instrument-drawings"),
]
DATA_ROOT = next((p for p in _KAGGLE_DATA_CANDIDATES if p.exists()), _KAGGLE_DATA_CANDIDATES[0])
WORK_DIR = Path("/kaggle/working")
OUT_DIR = WORK_DIR / REQ_NO.lower()
RESULT_PATH = WORK_DIR / "result.json"

YOLO_DIR = WORK_DIR / "yolo_dataset"
SEED = 42
VAL_RATIO = 0.2
EPOCHS = int(os.getenv("MH005_EPOCHS", "100"))
IMGSZ = int(os.getenv("MH005_IMGSZ", "640"))

CLASS_NAMES = [
    "ball_valve", "check_valve", "cock_valve", "flow_meter",
    "gate_valve", "globe_valve", "misc_instrument", "panel",
    "regulator", "safety_valve", "solenoid_valve",
]
NUM_CLASSES = len(CLASS_NAMES)

print(f"[INIT] req={REQ_NO} arch=yolov8m epochs={EPOCHS} imgsz={IMGSZ}", flush=True)
print(f"[DATA_ROOT] {DATA_ROOT} (exists={DATA_ROOT.exists()})", flush=True)

# Debug：列出 /kaggle/input/ 子目錄
_input_root = Path("/kaggle/input")
if _input_root.exists():
    _found = list(_input_root.rglob("*"))[:30]
    print(f"[DEBUG] /kaggle/input/ entries: {[str(p) for p in _found]}", flush=True)

# 找 images / labels 目錄（zip 上傳後可能有不同層次）
def find_dir(root: Path, name: str) -> Path:
    # 直接子目錄
    if (root / name).exists():
        return root / name
    # 一層 zip 解壓後可能有同名子目錄
    for sub in root.iterdir():
        if sub.is_dir() and (sub / name).exists():
            return sub / name
    return root / name  # fallback，讓後續邏輯自行處理

IMAGES_DIR = find_dir(DATA_ROOT, "raw")
LABELS_DIR = find_dir(DATA_ROOT, "labels_auto")
print(f"[DATA] images={IMAGES_DIR} exists={IMAGES_DIR.exists()}", flush=True)
print(f"[DATA] labels={LABELS_DIR} exists={LABELS_DIR.exists()}", flush=True)

# 建立 YOLO dataset 結構
for split in ["train", "val"]:
    (YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
all_images = sorted([f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in img_exts])
pairs = []
for img in all_images:
    label = LABELS_DIR / (img.stem + ".txt")
    if label.exists():
        pairs.append((img, label))

print(f"[PAIRS] {len(pairs)} image-label pairs found", flush=True)

random.seed(SEED)
random.shuffle(pairs)
n_val = int(len(pairs) * VAL_RATIO)
val_pairs = pairs[:n_val]
train_pairs = pairs[n_val:]
print(f"[SPLIT] train={len(train_pairs)} val={len(val_pairs)}", flush=True)

# Kaggle 不支援 symlink 跨 /kaggle/input → /kaggle/working，改用 copy
for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
    for img, label in split_pairs:
        dst_img = YOLO_DIR / split / "images" / img.name
        dst_lbl = YOLO_DIR / split / "labels" / label.name
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        if not dst_lbl.exists():
            shutil.copy2(label, dst_lbl)

yaml_path = YOLO_DIR / "dataset.yaml"
yaml_path.write_text(f"""path: {YOLO_DIR}
train: train/images
val: val/images

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
""")
print(f"[YAML] {yaml_path}", flush=True)

from ultralytics import YOLO

print(f"[MODEL] loading yolov8s.pt (pretrained COCO, smaller for stability)", flush=True)
model = YOLO("yolov8s.pt")

import traceback
import torch
print(f"[GPU] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}, mem={torch.cuda.get_device_properties(0).total_memory//1024**3}GB", flush=True)

t_start = time.time()
print(f"\n=== Training Start (epochs={EPOCHS}, imgsz={IMGSZ}, device={DEVICE}) ===", flush=True)
try:
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=16,
        amp=True,
        device=DEVICE,
        project=str(OUT_DIR),
        name="yolo_run",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        scale=0.5,
        flipud=0.1,
    )
except Exception as _train_err:
    err_txt = traceback.format_exc()
    print(f"[TRAIN ERROR] {_train_err}", flush=True)
    print(err_txt, flush=True)
    (WORK_DIR / "error.log").write_text(err_txt)
    raise
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] train elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

best_pt = OUT_DIR / "yolo_run" / "weights" / "best.pt"
if not best_pt.exists():
    best_pt = OUT_DIR / "yolo_run" / "weights" / "last.pt"

model_best = YOLO(str(best_pt))
metrics = model_best.val(data=str(yaml_path), imgsz=IMGSZ, device=DEVICE)
map50 = float(metrics.box.map50)
map50_95 = float(metrics.box.map)
print(f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}", flush=True)

per_class_map50 = {}
try:
    ap_class_index = metrics.box.ap_class_index
    ap_per_class = metrics.box.ap50
    for idx, ap_val in zip(ap_class_index, ap_per_class):
        class_name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else str(idx)
        per_class_map50[class_name] = round(float(ap_val), 4)
    print(f"[PER-CLASS mAP50] {per_class_map50}", flush=True)
except Exception as _pce:
    print(f"[WARN] per-class metrics 提取失敗: {_pce}", flush=True)

if map50 >= 0.92:
    verdict, tier = "pass", "達目標 mAP50>=0.92"
elif map50 >= 0.70:
    verdict, tier = "baseline", "達 baseline mAP50>=0.70"
else:
    verdict, tier = "fail", "未達 baseline"

result = {
    "req_no": REQ_NO,
    "run": "kaggle_v2_yolov8m",
    "arch": "yolov8m",
    "device": DEVICE,
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "train_seconds": train_seconds,
    "n_train": len(train_pairs),
    "n_val": len(val_pairs),
    "classes": CLASS_NAMES,
    "map50": round(map50, 4),
    "map50_95": round(map50_95, 4),
    "per_class_map50": per_class_map50,
    "verdict": verdict,
    "tier": tier,
    "target": "mAP50 >= 0.92",
    "baseline": "mAP50 >= 0.70",
    "best_path": str(best_pt),
    "augmentation": {"mosaic": 1.0, "hsv_h": 0.015, "scale": 0.5, "flipud": 0.1},
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)
