"""
MH-2026-005 儀器規格圖辨識 v3 — YOLOv8s fine-tune（本機 Mac MPS 版）

目標：mAP50 >= 0.70（baseline >= 0.60）
資料：270 張 instrument drawings + auto-labels (11 class PID symbols)
設備：Mac MPS
"""
import os
import json
import shutil
import random
import time
from pathlib import Path

REQ_NO = "MH-2026-005"
DATA_ROOT = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/instrument_drawings")
OUT_DIR = Path(f"/Users/yinghaowang/HurricaneCore/modelhub/training/{REQ_NO.lower()}")
RESULT_PATH = OUT_DIR / "result.json"

IMAGES_DIR = DATA_ROOT / "raw"
LABELS_DIR = DATA_ROOT / "labels_auto"
YOLO_DIR = OUT_DIR / "yolo_dataset"
SEED = 42
VAL_RATIO = 0.2
EPOCHS = int(os.getenv("MH005_EPOCHS", "50"))
IMGSZ = int(os.getenv("MH005_IMGSZ", "640"))

CLASS_NAMES = [
    "ball_valve", "check_valve", "cock_valve", "flow_meter",
    "gate_valve", "globe_valve", "misc_instrument", "panel",
    "regulator", "safety_valve", "solenoid_valve",
]
NUM_CLASSES = len(CLASS_NAMES)

print(f"[INIT] req={REQ_NO}")
print(f"[DATA] images={IMAGES_DIR} labels={LABELS_DIR}")

# --- 建立 YOLO dataset 結構 ---
for split in ["train", "val"]:
    (YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# 收集 image-label pairs
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
all_images = sorted([f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in img_exts])
pairs = []
for img in all_images:
    label = LABELS_DIR / (img.stem + ".txt")
    if label.exists():
        pairs.append((img, label))

print(f"[PAIRS] {len(pairs)} image-label pairs found")

# Split
random.seed(SEED)
random.shuffle(pairs)
n_val = int(len(pairs) * VAL_RATIO)
val_pairs = pairs[:n_val]
train_pairs = pairs[n_val:]
print(f"[SPLIT] train={len(train_pairs)} val={len(val_pairs)}")

# Symlink images + copy labels
for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
    for img, label in split_pairs:
        dst_img = YOLO_DIR / split / "images" / img.name
        dst_lbl = YOLO_DIR / split / "labels" / label.name
        if not dst_img.exists():
            os.symlink(img, dst_img)
        if not dst_lbl.exists():
            shutil.copy2(label, dst_lbl)

# Write dataset.yaml
yaml_path = YOLO_DIR / "dataset.yaml"
yaml_content = f"""path: {YOLO_DIR}
train: train/images
val: val/images

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
yaml_path.write_text(yaml_content)
print(f"[YAML] {yaml_path}")

# --- 訓練 ---
from ultralytics import YOLO

print(f"[MODEL] loading yolov8s.pt (pretrained COCO)")
model = YOLO("yolov8s.pt")

t_start = time.time()
print(f"\n=== Training Start (epochs={EPOCHS}, imgsz={IMGSZ}) ===", flush=True)
results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=16,
    device=DEVICE,  # MPS has shape mismatch bug in YOLOv8 validation
    project=str(OUT_DIR),
    name="yolo_run",
    exist_ok=True,
    patience=10,
    save=True,
    plots=True,
    verbose=True,
)
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] train elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

# --- 評估 ---
print("\n=== Validation ===", flush=True)
best_pt = OUT_DIR / "yolo_run" / "weights" / "best.pt"
if not best_pt.exists():
    best_pt = OUT_DIR / "yolo_run" / "weights" / "last.pt"

model_best = YOLO(str(best_pt))
metrics = model_best.val(data=str(yaml_path), imgsz=IMGSZ, device=DEVICE)

map50 = float(metrics.box.map50)
map50_95 = float(metrics.box.map)
print(f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}", flush=True)

# 結論
if map50 >= 0.70:
    verdict, tier = "pass", "達標"
elif map50 >= 0.60:
    verdict, tier = "baseline", "baseline 可交付"
else:
    verdict, tier = "fail", "未達 baseline"

result = {
    "req_no": REQ_NO,
    "device": "mps",
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "train_seconds": train_seconds,
    "n_train": len(train_pairs),
    "n_val": len(val_pairs),
    "classes": CLASS_NAMES,
    "map50": round(map50, 4),
    "map50_95": round(map50_95, 4),
    "verdict": verdict,
    "tier": tier,
    "target": "mAP50 >= 0.70",
    "baseline": "mAP50 >= 0.60",
    "best_path": str(best_pt),
}
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)
