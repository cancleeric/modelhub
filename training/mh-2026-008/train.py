"""
MH-2026-008 多視圖邊界偵測 — YOLOv8s fine-tune（本機 CPU 版）

目標：mAP50 >= 0.70（baseline >= 0.60）
資料：212 張 multiview_drawings + auto-labels (11 class PID)
設備：CPU
"""
import os, json, shutil, random, time
from pathlib import Path

REQ_NO = "MH-2026-008"
DATA_ROOT = Path("/Users/yinghaowang/HurricaneCore/docker-data/modelhub/datasets/multiview_drawings")
OUT_DIR = Path(f"/Users/yinghaowang/HurricaneCore/modelhub/training/{REQ_NO.lower()}")
RESULT_PATH = OUT_DIR / "result.json"
IMAGES_DIR = DATA_ROOT / "raw"
LABELS_DIR = DATA_ROOT / "labels_auto"
YOLO_DIR = OUT_DIR / "yolo_dataset"
SEED = 42
VAL_RATIO = 0.2
EPOCHS = int(os.getenv("MH008_EPOCHS", "50"))
IMGSZ = int(os.getenv("MH008_IMGSZ", "640"))
CLASS_NAMES = ["ball_valve","check_valve","cock_valve","flow_meter","gate_valve","globe_valve","misc_instrument","panel","regulator","safety_valve","solenoid_valve"]

print(f"[INIT] req={REQ_NO}", flush=True)
for split in ["train", "val"]:
    (YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
all_images = sorted([f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in img_exts])
pairs = [(img, LABELS_DIR / (img.stem + ".txt")) for img in all_images if (LABELS_DIR / (img.stem + ".txt")).exists()]
print(f"[PAIRS] {len(pairs)}", flush=True)

random.seed(SEED)
random.shuffle(pairs)
n_val = int(len(pairs) * VAL_RATIO)
val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
print(f"[SPLIT] train={len(train_pairs)} val={len(val_pairs)}", flush=True)

for split, sp in [("train", train_pairs), ("val", val_pairs)]:
    for img, label in sp:
        dst_img = YOLO_DIR / split / "images" / img.name
        dst_lbl = YOLO_DIR / split / "labels" / label.name
        if not dst_img.exists(): os.symlink(img, dst_img)
        if not dst_lbl.exists(): shutil.copy2(label, dst_lbl)

yaml_path = YOLO_DIR / "dataset.yaml"
yaml_path.write_text(f"path: {YOLO_DIR}\ntrain: train/images\nval: val/images\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")

from ultralytics import YOLO
model = YOLO("yolov8s.pt")
t_start = time.time()
print(f"\n=== Training Start (epochs={EPOCHS}) ===", flush=True)
model.train(data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=16, device="cpu",
            project=str(OUT_DIR), name="yolo_run", exist_ok=True, patience=10, save=True, verbose=True)
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

best_pt = OUT_DIR / "yolo_run" / "weights" / "best.pt"
if not best_pt.exists(): best_pt = OUT_DIR / "yolo_run" / "weights" / "last.pt"
model_best = YOLO(str(best_pt))
metrics = model_best.val(data=str(yaml_path), imgsz=IMGSZ, device="cpu")
map50 = float(metrics.box.map50)
map50_95 = float(metrics.box.map)
print(f"mAP50={map50:.4f} mAP50-95={map50_95:.4f}", flush=True)

if map50 >= 0.70: verdict, tier = "pass", "達標"
elif map50 >= 0.60: verdict, tier = "baseline", "baseline 可交付"
else: verdict, tier = "fail", "未達 baseline"

RESULT_PATH.write_text(json.dumps({"req_no": REQ_NO, "device": "cpu", "epochs": EPOCHS,
    "train_seconds": train_seconds, "n_train": len(train_pairs), "n_val": len(val_pairs),
    "map50": round(map50, 4), "map50_95": round(map50_95, 4), "verdict": verdict, "tier": tier,
    "best_path": str(best_pt)}, ensure_ascii=False, indent=2))
print(f"\n結論：{tier}", flush=True)

# --- Sprint 8.2: 自動回寫 ModelHub DB ---
import sys as _sys
_sys.path.insert(0, str(OUT_DIR.parent))
try:
    from modelhub_report import report_result
    report_result(
        req_no=REQ_NO,
        passed=(verdict in ("pass", "baseline")),
        metrics={"map50": round(map50, 4), "map50_95": round(map50_95, 4), "epochs": EPOCHS},
        model_path=str(best_pt),
        notes=f"train.py 自動回寫: {tier}",
    )
except Exception as _e:
    print(f"[WARN] modelhub_report 回寫失敗（不中斷）: {_e}", flush=True)
