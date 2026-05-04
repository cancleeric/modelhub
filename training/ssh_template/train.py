"""
ssh_template/train.py — SSH 遠端 GPU 訓練腳本模板

使用方式：
  複製本目錄到新的訓練任務，修改下方 CONFIG 區段後上傳到遠端 SSH 主機執行。

SSHLauncher.submit_job() 會透過 rsync 上傳 dataset + config.json，
並以 nohup 背景執行本腳本，將輸出 redirect 到 ~/modelhub-jobs/<req_no>/train.log。

result.json 規格（ssh_poller 讀取）：
  - map50, map50_95         必填，mAP 指標
  - verdict                 必填，"pass" / "baseline" / "fail"
  - dataset_snapshot_id     選填，dataset 版本識別（SHA256 / Kaggle dataset id）
  - train_commit_hash       選填，訓練腳本 Git commit SHA（40 chars）
  - hyperparams             選填，完整 hyperparams dict

注意：
  - 遠端主機需有 CUDA GPU + ultralytics 已安裝
  - SSH key 認證，不使用密碼
  - 執行完後在 ~/modelhub-jobs/<req_no>/ 下建立 job.done 或 job.failed 檔
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIG — 修改此區段
# ---------------------------------------------------------------------------
REQ_NO = os.environ.get("MH_REQ_NO", "UNKNOWN")
ARCH = os.environ.get("MH_ARCH", "yolov8m.pt")
EPOCHS = int(os.environ.get("MH_EPOCHS", "50"))
IMGSZ = int(os.environ.get("MH_IMGSZ", "640"))
BATCH = int(os.environ.get("MH_BATCH", "16"))
DEVICE = os.environ.get("MH_DEVICE", "0")           # "0" = GPU 0，"cpu" = CPU
MAP50_TARGET = float(os.environ.get("MH_MAP50_TARGET", "0.70"))
MAP50_BASELINE = float(os.environ.get("MH_MAP50_BASELINE", "0.60"))
CLASS_NAMES: list = []                               # 由 config.json 讀取或在此設定
# ---------------------------------------------------------------------------

JOB_DIR = Path(f"~/modelhub-jobs/{REQ_NO}").expanduser()
DATASET_DIR = JOB_DIR / "dataset"
RESULT_PATH = JOB_DIR / "result.json"
BEST_PT_SRC = JOB_DIR / "yolo_run" / "weights" / "best.pt"
BEST_PT_DEST = JOB_DIR / "best.pt"


def _get_git_commit() -> str:
    """取得目前腳本所在 Git commit SHA"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).parent),
            timeout=5,
        ).decode().strip()
    except Exception:
        return ""


def _get_dataset_snapshot_id() -> str:
    """
    取得 dataset 版本識別。
    優先讀 dataset/snapshot_id.txt（由 SSHLauncher rsync 時附帶），
    fallback 計算 dataset 目錄的 SHA256 摘要（僅用目錄樹，不讀每個檔案）。
    """
    snapshot_file = DATASET_DIR / "snapshot_id.txt"
    if snapshot_file.exists():
        return snapshot_file.read_text().strip()
    # fallback：用目錄樹路徑作為識別
    try:
        import hashlib
        tree = sorted(str(p.relative_to(DATASET_DIR)) for p in DATASET_DIR.rglob("*") if p.is_file())
        h = hashlib.sha256("\n".join(tree).encode()).hexdigest()[:16]
        return f"tree-sha256-{h}"
    except Exception:
        return ""


def main():
    done_file = JOB_DIR / "job.done"
    failed_file = JOB_DIR / "job.failed"

    # 讀取 config.json（SSHLauncher 寫入）
    config_path = JOB_DIR / "config.json"
    config: dict = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            print(f"[WARN] config.json 讀取失敗: {e}", flush=True)

    # 讀取 class_names
    global CLASS_NAMES
    CLASS_NAMES = config.get("class_names", CLASS_NAMES) or []
    if not CLASS_NAMES:
        # 從 dataset.yaml 自動推測
        dataset_yaml = DATASET_DIR / "dataset.yaml"
        if dataset_yaml.exists():
            try:
                import yaml  # type: ignore
                with open(dataset_yaml) as f:
                    y = yaml.safe_load(f)
                CLASS_NAMES = y.get("names", [])
            except Exception:
                pass

    # 覆寫 config 中的訓練參數
    req_no_cfg = config.get("req_no", REQ_NO)
    arch = config.get("arch", ARCH)
    epochs = int(config.get("epochs", EPOCHS))
    imgsz = int(config.get("imgsz", IMGSZ))
    batch = int(config.get("batch", BATCH))
    device = config.get("device", DEVICE)

    print(f"[SSH Train] req_no={req_no_cfg} arch={arch} epochs={epochs} device={device}", flush=True)

    t0 = time.time()
    try:
        from ultralytics import YOLO

        # 確保 YOLO 輸出目錄
        run_dir = JOB_DIR / "yolo_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(arch)
        model.train(
            data=str(DATASET_DIR / "dataset.yaml"),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(JOB_DIR),
            name="yolo_run",
            save=True,
            plots=True,
            verbose=True,
            exist_ok=True,
        )

        train_seconds = int(time.time() - t0)

        # 評估
        metrics = model.val(data=str(DATASET_DIR / "dataset.yaml"), imgsz=imgsz, device=device)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        print(f"[SSH Train] mAP50={map50:.4f} mAP50-95={map50_95:.4f}", flush=True)

        # per-class mAP50
        per_class_map50: dict = {}
        try:
            ap_class_index = metrics.box.ap_class_index
            ap_per_cls = metrics.box.ap50
            for idx, ap_val in zip(ap_class_index, ap_per_cls):
                name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else str(idx)
                per_class_map50[name] = round(float(ap_val), 4)
        except Exception as e:
            print(f"[WARN] per-class metrics 提取失敗: {e}", flush=True)

        if map50 >= MAP50_TARGET:
            verdict = "pass"
        elif map50 >= MAP50_BASELINE:
            verdict = "baseline"
        else:
            verdict = "fail"

        # 複製 best.pt 到 job 根目錄（供 SSHLauncher.download_output 下載）
        if BEST_PT_SRC.exists():
            import shutil
            shutil.copy2(BEST_PT_SRC, BEST_PT_DEST)

        result = {
            "req_no": req_no_cfg,
            "device": device,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "train_seconds": train_seconds,
            "classes": CLASS_NAMES,
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "per_class_map50": per_class_map50,
            "verdict": verdict,
            "target": f"mAP50 >= {MAP50_TARGET}",
            "baseline": f"mAP50 >= {MAP50_BASELINE}",
            "best_path": str(BEST_PT_DEST),
            # 追溯性欄位（ssh_poller 讀取並填入 ModelVersion）
            "dataset_snapshot_id": _get_dataset_snapshot_id(),
            "train_commit_hash": _get_git_commit(),
            "hyperparams": {
                "arch": arch,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
            },
        }

        RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(f"\n結論：{verdict}", flush=True)

        # 標記完成
        done_file.write_text("done")
        sys.exit(0)

    except Exception as e:
        train_seconds = int(time.time() - t0)
        print(f"[SSH Train] FAILED after {train_seconds}s: {e}", flush=True)
        import traceback
        traceback.print_exc()

        # 寫入失敗摘要到 result.json（讓 poller 可讀取部分資訊）
        try:
            RESULT_PATH.write_text(json.dumps({
                "req_no": REQ_NO,
                "verdict": "fail",
                "error": str(e),
                "train_seconds": train_seconds,
            }, ensure_ascii=False, indent=2))
        except Exception:
            pass

        # 標記失敗
        failed_file.write_text(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
