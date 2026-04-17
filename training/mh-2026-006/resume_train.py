"""
MH-2026-006 Resume 重訓腳本（Sprint 8.4）

從 yolo_run/weights/best.pt 繼續訓練 20 epochs，
完成後寫 result_v2.json 並呼叫 modelhub_report（若可用）。
"""
import json
import sys
import time
from pathlib import Path

OUT_DIR = Path("/Users/yinghaowang/HurricaneCore/modelhub/training/mh-2026-006")
RESUME_PT = OUT_DIR / "yolo_run" / "weights" / "best.pt"
YAML_PATH = OUT_DIR / "yolo_dataset" / "dataset.yaml"
RESULT_V2_PATH = OUT_DIR / "result_v2.json"
REQ_NO = "MH-2026-006"

EPOCHS = 20
IMGSZ = 640

print(f"[8.4] Resume training {REQ_NO} from {RESUME_PT}", flush=True)
print(f"[8.4] epochs={EPOCHS} imgsz={IMGSZ} device=cpu", flush=True)

if not RESUME_PT.exists():
    print(f"[ERROR] {RESUME_PT} not found, aborting", flush=True)
    sys.exit(1)

if not YAML_PATH.exists():
    print(f"[ERROR] {YAML_PATH} not found, aborting", flush=True)
    sys.exit(1)

from ultralytics import YOLO

t_start = time.time()

# Resume 訓練：載入已訓練的 best.pt，繼續 fine-tune
model = YOLO(str(RESUME_PT))
print(f"\n=== Resume Training Start ===", flush=True)
results = model.train(
    data=str(YAML_PATH),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=16,
    device="cpu",
    project=str(OUT_DIR),
    name="resume_run",
    exist_ok=True,
    patience=10,
    save=True,
    plots=True,
    verbose=True,
)
train_seconds = int(time.time() - t_start)
print(f"\n[DONE] resume elapsed {train_seconds}s = {train_seconds/3600:.2f}h", flush=True)

# --- Eval ---
best_pt_v2 = OUT_DIR / "resume_run" / "weights" / "best.pt"
if not best_pt_v2.exists():
    best_pt_v2 = OUT_DIR / "resume_run" / "weights" / "last.pt"
if not best_pt_v2.exists():
    best_pt_v2 = RESUME_PT

model_best = YOLO(str(best_pt_v2))
metrics = model_best.val(data=str(YAML_PATH), imgsz=IMGSZ, device="cpu")
map50 = float(metrics.box.map50)
map50_95 = float(metrics.box.map)
print(f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}", flush=True)

if map50 >= 0.70:
    verdict, tier = "pass", "達標"
elif map50 >= 0.60:
    verdict, tier = "baseline", "baseline 可交付"
else:
    verdict, tier = "fail", "未達 baseline"

result_v2 = {
    "req_no": REQ_NO,
    "run": "resume_v2",
    "base_checkpoint": str(RESUME_PT),
    "device": "cpu",
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "train_seconds": train_seconds,
    "map50": round(map50, 4),
    "map50_95": round(map50_95, 4),
    "verdict": verdict,
    "tier": tier,
    "target": "mAP50 >= 0.70",
    "baseline": "mAP50 >= 0.60",
    "best_path": str(best_pt_v2),
}
RESULT_V2_PATH.write_text(json.dumps(result_v2, ensure_ascii=False, indent=2))
print(json.dumps(result_v2, ensure_ascii=False, indent=2), flush=True)
print(f"\n結論：{tier}", flush=True)

# --- 回寫 ModelHub DB（8.2 完工後整合）---
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from modelhub_report import report_result
    passed = verdict in ("pass", "baseline")
    report_result(
        req_no=REQ_NO,
        passed=passed,
        metrics={"map50": round(map50, 4), "map50_95": round(map50_95, 4), "epochs": EPOCHS, "run": "resume_v2"},
        model_path=str(best_pt_v2),
        notes=f"Sprint 8.4 resume 20ep: {tier}",
    )
except Exception as e:
    print(f"[WARN] modelhub_report 回寫失敗（不中斷）: {e}", flush=True)
