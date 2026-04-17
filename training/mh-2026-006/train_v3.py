"""
MH-2026-006 第三輪訓練腳本（Sprint 13 P1-B）

基於 resume_run/weights/best.pt 繼續 resume 訓練。
改善項：
- epochs=100（原 resume_train.py 僅 20）
- patience=20
- augmentation 加強：mosaic=1.0、flipud=0.5、scale=0.5
- 使用 yolov8m.pt 架構（但從 resume_run/best.pt resume）
- 整合 Sprint 13 P0-A 修正後的 modelhub_report 回寫（需設定 MODELHUB_API_KEY）

用法：
    python3 training/mh-2026-006/train_v3.py

背景執行（推薦）：
    nohup python3 training/mh-2026-006/train_v3.py \\
        > modelhub/logs/mh-2026-006-v3-$(date +%Y%m%d-%H%M%S).log 2>&1 &
"""
import json
import sys
import time
from pathlib import Path

OUT_DIR = Path("/Users/yinghaowang/HurricaneCore/modelhub/training/mh-2026-006")
# 從 resume_run（第二輪）的 best.pt 繼續
RESUME_PT = OUT_DIR / "resume_run" / "weights" / "best.pt"
# fallback：若 resume_run 不存在，從原始 yolo_run 的 best.pt
if not RESUME_PT.exists():
    RESUME_PT = OUT_DIR / "yolo_run" / "weights" / "best.pt"

YAML_PATH = OUT_DIR / "yolo_dataset" / "dataset.yaml"
RESULT_V3_PATH = OUT_DIR / "result_v3.json"
REQ_NO = "MH-2026-006"
CLASS_NAMES = ["text"]

EPOCHS = 100
IMGSZ = 640
PATIENCE = 20

print(f"[INIT] Sprint 13 P1-B: {REQ_NO} train_v3", flush=True)
print(f"[INIT] resume from: {RESUME_PT}", flush=True)
print(f"[INIT] epochs={EPOCHS} imgsz={IMGSZ} patience={PATIENCE}", flush=True)

if not RESUME_PT.exists():
    print(f"[ERROR] checkpoint 不存在：{RESUME_PT}", file=sys.stderr, flush=True)
    sys.exit(1)

if not YAML_PATH.exists():
    print(f"[ERROR] dataset.yaml 不存在：{YAML_PATH}", file=sys.stderr, flush=True)
    sys.exit(1)

from ultralytics import YOLO

# 從 best.pt resume（載入已訓練權重繼續 fine-tune，架構已固定為 yolov8s）
model = YOLO(str(RESUME_PT))

t_start = time.time()
print(f"\n=== V3 Training Start (epochs={EPOCHS}, mosaic=1.0, flipud=0.5, scale=0.5) ===", flush=True)
model.train(
    data=str(YAML_PATH),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=16,
    device="cpu",
    project=str(OUT_DIR),
    name="v3_run",
    exist_ok=True,
    patience=PATIENCE,
    save=True,
    plots=True,
    verbose=True,
    # 加強 augmentation
    mosaic=1.0,
    flipud=0.5,
    scale=0.5,
)
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] v3 elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

# --- Eval ---
best_pt_v3 = OUT_DIR / "v3_run" / "weights" / "best.pt"
if not best_pt_v3.exists():
    best_pt_v3 = OUT_DIR / "v3_run" / "weights" / "last.pt"
if not best_pt_v3.exists():
    best_pt_v3 = RESUME_PT

model_best = YOLO(str(best_pt_v3))
val_metrics = model_best.val(data=str(YAML_PATH), imgsz=IMGSZ, device="cpu")
map50 = float(val_metrics.box.map50)
map50_95 = float(val_metrics.box.map)
print(f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}", flush=True)

# per-class AP50
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

result_v3 = {
    "req_no": REQ_NO,
    "run": "v3",
    "base_checkpoint": str(RESUME_PT),
    "device": "cpu",
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "patience": PATIENCE,
    "augmentation": {"mosaic": 1.0, "flipud": 0.5, "scale": 0.5},
    "train_seconds": train_seconds,
    "map50": round(map50, 4),
    "map50_95": round(map50_95, 4),
    "per_class_map50": per_class_map50,
    "verdict": verdict,
    "tier": tier,
    "target": "mAP50 >= 0.70",
    "baseline": "mAP50 >= 0.60",
    "best_path": str(best_pt_v3),
}
RESULT_V3_PATH.write_text(json.dumps(result_v3, ensure_ascii=False, indent=2))
print(json.dumps(result_v3, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)

# --- Sprint 13 P0-A 修正後的回寫（需設定 MODELHUB_API_KEY）---
sys.path.insert(0, str(OUT_DIR.parent))
try:
    from modelhub_report import report_result
    report_result(
        req_no=REQ_NO,
        passed=(verdict in ("pass", "baseline")),
        metrics={"map50": round(map50, 4), "map50_95": round(map50_95, 4), "epochs": EPOCHS, "run": "v3"},
        model_path=str(best_pt_v3),
        notes=f"Sprint 13 P1-B train_v3 (100ep, mosaic+flipud+scale): {tier}",
        per_class_metrics=per_class_map50 if per_class_map50 else None,
    )
except Exception as _e:
    print(f"[WARN] modelhub_report 回寫失敗（不中斷）: {_e}", flush=True)
