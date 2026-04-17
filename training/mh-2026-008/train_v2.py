"""
MH-2026-008 第二輪訓練腳本（Sprint 13 P1-C 步驟 2）

使用原始 212 張 + augmented_dataset 合併資料。
改善項：
- 改用 yolov8m.pt（比 yolov8s.pt 更大，更適合 11 類）
- epochs=100，patience=20
- 整合 per-class metrics 輸出（P2-A）
- 整合 P0-A 修正後的回寫（需設定 MODELHUB_API_KEY）

前置條件：
    先執行 augment_rare_classes.py 產生 augmented_dataset/

用法：
    python3 training/mh-2026-008/train_v2.py

背景執行（推薦）：
    nohup python3 training/mh-2026-008/train_v2.py \\
        > modelhub/logs/mh-2026-008-v2-$(date +%Y%m%d-%H%M%S).log 2>&1 &
"""
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

REQ_NO = "MH-2026-008"
DATA_ROOT = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/multiview_drawings")
OUT_DIR = Path(f"/Users/yinghaowang/HurricaneCore/modelhub/training/{REQ_NO.lower()}")
AUG_DIR = OUT_DIR / "augmented_dataset"
RESULT_PATH = OUT_DIR / "result_v2.json"
IMAGES_DIR = DATA_ROOT / "raw"
LABELS_DIR = DATA_ROOT / "labels_auto"
YOLO_V2_DIR = OUT_DIR / "yolo_v2_dataset"

SEED = 42
VAL_RATIO = 0.2
EPOCHS = int(os.getenv("MH008_V2_EPOCHS", "100"))
IMGSZ = int(os.getenv("MH008_V2_IMGSZ", "640"))
PATIENCE = 20

CLASS_NAMES = [
    "ball_valve", "check_valve", "cock_valve", "flow_meter", "gate_valve",
    "globe_valve", "misc_instrument", "panel", "regulator", "safety_valve",
    "solenoid_valve",
]

print(f"[INIT] Sprint 13 P1-C: {REQ_NO} train_v2", flush=True)
print(f"[INIT] epochs={EPOCHS} imgsz={IMGSZ} patience={PATIENCE} arch=yolov8m", flush=True)

# ---------------------------------------------------------------------------
# 建立合併資料集（原始 + augmented）
# ---------------------------------------------------------------------------
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 建立 yolo_v2_dataset 目錄
for split in ["train", "val"]:
    (YOLO_V2_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_V2_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# 原始資料
all_images = sorted([f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in img_exts])
pairs = [
    (img, LABELS_DIR / (img.stem + ".txt"))
    for img in all_images
    if (LABELS_DIR / (img.stem + ".txt")).exists()
]

# augmented 資料（若存在）
aug_images_dir = AUG_DIR / "images"
aug_labels_dir = AUG_DIR / "labels"
aug_pairs = []
if aug_images_dir.exists():
    aug_imgs = sorted([f for f in aug_images_dir.iterdir() if f.suffix.lower() in img_exts])
    aug_pairs = [
        (img, aug_labels_dir / (img.stem + ".txt"))
        for img in aug_imgs
        if (aug_labels_dir / (img.stem + ".txt")).exists()
    ]
    print(f"[AUGMENTED] 找到 {len(aug_pairs)} 張 augmented 圖片", flush=True)
else:
    print(f"[WARN] augmented_dataset 不存在，僅使用原始資料。建議先執行 augment_rare_classes.py", flush=True)

all_pairs = pairs + aug_pairs
print(f"[PAIRS] 原始={len(pairs)} augmented={len(aug_pairs)} total={len(all_pairs)}", flush=True)

# 只對原始資料做 train/val split（augmented 全進 train，避免 data leakage）
random.seed(SEED)
random.shuffle(pairs)
n_val = int(len(pairs) * VAL_RATIO)
val_pairs = pairs[:n_val]
train_pairs_orig = pairs[n_val:]
train_pairs = train_pairs_orig + aug_pairs

print(f"[SPLIT] train={len(train_pairs)} (orig={len(train_pairs_orig)} aug={len(aug_pairs)}) val={len(val_pairs)}", flush=True)

# Symlink/copy 到 yolo_v2_dataset
for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
    for img, label in split_pairs:
        dst_img = YOLO_V2_DIR / split / "images" / img.name
        dst_lbl = YOLO_V2_DIR / split / "labels" / label.name
        if not dst_img.exists():
            try:
                os.symlink(img, dst_img)
            except Exception:
                shutil.copy2(img, dst_img)
        if label.exists() and not dst_lbl.exists():
            shutil.copy2(label, dst_lbl)

yaml_path = YOLO_V2_DIR / "dataset.yaml"
yaml_path.write_text(
    f"path: {YOLO_V2_DIR}\n"
    f"train: train/images\n"
    f"val: val/images\n\n"
    f"nc: {len(CLASS_NAMES)}\n"
    f"names: {CLASS_NAMES}\n"
)

# ---------------------------------------------------------------------------
# Train（yolov8m.pt）
# ---------------------------------------------------------------------------
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
t_start = time.time()
print(f"\n=== V2 Training Start (yolov8m, epochs={EPOCHS}, patience={PATIENCE}) ===", flush=True)
model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=8,        # m 模型記憶體需求更大，batch 縮小
    device="cpu",
    project=str(OUT_DIR),
    name="v2_run",
    exist_ok=True,
    patience=PATIENCE,
    save=True,
    plots=True,
    verbose=True,
)
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] v2 elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

# ---------------------------------------------------------------------------
# Eval + per-class metrics
# ---------------------------------------------------------------------------
best_pt_v2 = OUT_DIR / "v2_run" / "weights" / "best.pt"
if not best_pt_v2.exists():
    best_pt_v2 = OUT_DIR / "v2_run" / "weights" / "last.pt"

model_best = YOLO(str(best_pt_v2))
val_metrics = model_best.val(data=str(yaml_path), imgsz=IMGSZ, device="cpu")
map50 = float(val_metrics.box.map50)
map50_95 = float(val_metrics.box.map)
print(f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}", flush=True)

# per-class AP50（Sprint 13 P2-A）
per_class_map50 = {}
try:
    ap_class_index = val_metrics.box.ap_class_index
    ap_per_cls = val_metrics.box.ap50
    for idx, ap_val in zip(ap_class_index, ap_per_cls):
        name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else str(idx)
        per_class_map50[name] = round(float(ap_val), 4)
    print(f"[PER-CLASS mAP50] {per_class_map50}", flush=True)
except Exception as _pce:
    print(f"[WARN] per-class metrics 提取失敗: {_pce}", flush=True)

if map50 >= 0.70:
    verdict, tier = "pass", "達標"
elif map50 >= 0.60:
    verdict, tier = "baseline", "baseline 可交付"
else:
    verdict, tier = "fail", "未達 baseline"

result_v2 = {
    "req_no": REQ_NO,
    "run": "v2",
    "arch": "yolov8m",
    "device": "cpu",
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "patience": PATIENCE,
    "n_train_orig": len(train_pairs_orig),
    "n_train_aug": len(aug_pairs),
    "n_train_total": len(train_pairs),
    "n_val": len(val_pairs),
    "train_seconds": train_seconds,
    "map50": round(map50, 4),
    "map50_95": round(map50_95, 4),
    "per_class_map50": per_class_map50,
    "verdict": verdict,
    "tier": tier,
    "target": "mAP50 >= 0.70",
    "baseline": "mAP50 >= 0.60",
    "best_path": str(best_pt_v2),
}
RESULT_PATH.write_text(json.dumps(result_v2, ensure_ascii=False, indent=2))
print(json.dumps(result_v2, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)

# --- Sprint 13 P0-A 修正後的回寫（需設定 MODELHUB_API_KEY）---
sys.path.insert(0, str(OUT_DIR.parent))
try:
    from modelhub_report import report_result
    report_result(
        req_no=REQ_NO,
        passed=(verdict in ("pass", "baseline")),
        metrics={"map50": round(map50, 4), "map50_95": round(map50_95, 4), "epochs": EPOCHS, "run": "v2"},
        model_path=str(best_pt_v2),
        notes=f"Sprint 13 P1-C train_v2 (yolov8m, aug+orig): {tier}",
        per_class_metrics=per_class_map50 if per_class_map50 else None,
    )
except Exception as _e:
    print(f"[WARN] modelhub_report 回寫失敗（不中斷）: {_e}", flush=True)
