"""
base_trainer.py — Sprint 13 P3-A: 共用 BaseTrainer 工具函式

提供各訓練腳本共用的邏輯：
- prepare_dataset_split()   — 資料集分割 + YOLO 目錄結構建立
- build_yolo_config()       — 產生 YOLO training kwargs dict
- eval_and_write_result()   — 評估模型並寫 result.json
- report_to_modelhub()      — 呼叫 modelhub_report 回寫 API

目前已整合：mh-2026-006、mh-2026-008
其他訓練腳本待整合（TODO 註記見各 train.py）
"""
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def prepare_dataset_split(
    images_dir: Path,
    labels_dir: Path,
    yolo_dir: Path,
    class_names: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], Path]:
    """
    掃描 images_dir / labels_dir，分割 train/val，建立 YOLO 目錄結構並寫 dataset.yaml。

    Returns:
        (train_pairs, val_pairs, yaml_path)
        train_pairs: [(img_path, label_path), ...]
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for split in ["train", "val"]:
        (yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    all_images = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in img_exts])
    pairs = [
        (img, labels_dir / (img.stem + ".txt"))
        for img in all_images
        if (labels_dir / (img.stem + ".txt")).exists()
    ]

    random.seed(seed)
    random.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img, label in split_pairs:
            dst_img = yolo_dir / split / "images" / img.name
            dst_lbl = yolo_dir / split / "labels" / label.name
            if not dst_img.exists():
                os.symlink(img, dst_img)
            if not dst_lbl.exists():
                shutil.copy2(label, dst_lbl)

    yaml_path = yolo_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {yolo_dir}\n"
        f"train: train/images\n"
        f"val: val/images\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    return train_pairs, val_pairs, yaml_path


# ---------------------------------------------------------------------------
# YOLO config builder
# ---------------------------------------------------------------------------

def build_yolo_config(
    arch: str = "yolov8s.pt",
    epochs: int = 50,
    patience: int = 10,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    **augmentation_kwargs,
) -> Dict:
    """
    產生 YOLO model.train() 的 kwargs dict（不含 data/project/name）。

    augmentation_kwargs 可傳：mosaic, flipud, scale, fliplr, degrees 等
    """
    cfg = {
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "patience": patience,
        "save": True,
        "plots": True,
        "verbose": True,
    }
    cfg.update(augmentation_kwargs)
    return cfg


# ---------------------------------------------------------------------------
# Eval + write result.json
# ---------------------------------------------------------------------------

def eval_and_write_result(
    model,
    yaml_path: Path,
    result_path: Path,
    req_no: str,
    class_names: List[str],
    train_pairs: list,
    val_pairs: list,
    best_pt: Path,
    train_seconds: int,
    epochs: int,
    imgsz: int,
    device: Optional[str] = None,
    map50_target: float = 0.70,
    map50_baseline: float = 0.60,
    extra_fields: Optional[Dict] = None,
    per_class: bool = True,
    dataset_snapshot_id: Optional[str] = None,
    train_commit_hash: Optional[str] = None,
    hyperparams: Optional[Dict] = None,
) -> Dict:
    """
    對 model 執行 val()，計算 mAP50/mAP50-95 及 per-class AP50，
    判斷 verdict，寫 result.json，並回傳 result dict。

    P2-17: device 必須由 caller 明確傳入，不接受 None（本機 caller 請傳 "mps"）

    追溯性欄位（供 ssh_poller 填入 ModelVersion）：
    - dataset_snapshot_id: dataset 版本識別（如 Kaggle dataset id / SHA256 hash）
    - train_commit_hash:   訓練腳本所在 Git commit SHA（40 chars）
                          可由訓練腳本自動取得：
                          subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    - hyperparams:         完整 hyperparams dict（epochs/batch/imgsz/arch 等）
                          若不傳則由此函式自動組合基本欄位
    """
    if device is None:
        raise ValueError("device 必須明確傳入（例：'mps' 或 '0'），不可為 None")
    metrics = model.val(data=str(yaml_path), imgsz=imgsz, device=device)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    print(f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}", flush=True)

    per_class_map50: Dict[str, float] = {}
    if per_class:
        try:
            ap_class_index = metrics.box.ap_class_index
            ap_per_cls = metrics.box.ap50
            for idx, ap_val in zip(ap_class_index, ap_per_cls):
                name = class_names[int(idx)] if int(idx) < len(class_names) else str(idx)
                per_class_map50[name] = round(float(ap_val), 4)
            print(f"[PER-CLASS mAP50] {per_class_map50}", flush=True)
        except Exception as e:
            print(f"[WARN] per-class metrics 提取失敗: {e}", flush=True)

    if map50 >= map50_target:
        verdict, tier = "pass", "達標"
    elif map50 >= map50_baseline:
        verdict, tier = "baseline", "baseline 可交付"
    else:
        verdict, tier = "fail", "未達 baseline"

    # 自動組合基本 hyperparams（若 caller 未傳）
    _hyperparams = hyperparams or {
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
    }

    result = {
        "req_no": req_no,
        "device": device,
        "epochs": epochs,
        "imgsz": imgsz,
        "train_seconds": train_seconds,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "classes": class_names,
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "per_class_map50": per_class_map50,
        "verdict": verdict,
        "tier": tier,
        "target": f"mAP50 >= {map50_target}",
        "baseline": f"mAP50 >= {map50_baseline}",
        "best_path": str(best_pt),
        # 追溯性欄位（ssh_poller 讀取並填入 ModelVersion）
        "dataset_snapshot_id": dataset_snapshot_id,
        "train_commit_hash": train_commit_hash,
        "hyperparams": _hyperparams,
    }
    if extra_fields:
        result.update(extra_fields)

    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # Kaggle 環境：同步寫一份到 /kaggle/working/result.json（供 kaggle kernels output 抓取）
    _kaggle_working = Path("/kaggle/working")
    if _kaggle_working.exists() and result_path.resolve() != (_kaggle_working / "result.json").resolve():
        try:
            (_kaggle_working / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as _e:
            print(f"[WARN] 無法寫入 /kaggle/working/result.json: {_e}", flush=True)

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"\n結論：{tier}", flush=True)
    # 標記行：kaggle_poller 從 log 解析指標用（確保 result.json 無法下載時仍可回填）
    print("##RESULT_JSON##:" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


# ---------------------------------------------------------------------------
# Report to ModelHub
# ---------------------------------------------------------------------------

def report_to_modelhub(
    req_no: str,
    result: Dict,
    model_path: Optional[Path] = None,
    notes: str = "",
) -> None:
    """
    呼叫 modelhub_report.report_result()，靜默失敗不中斷訓練流程。
    """
    import sys
    training_dir = Path(__file__).parent
    sys.path.insert(0, str(training_dir))
    try:
        from modelhub_report import report_result
        verdict = result.get("verdict", "fail")
        passed = verdict in ("pass", "baseline")
        metrics = {
            "map50": result.get("map50"),
            "map50_95": result.get("map50_95"),
            "epochs": result.get("epochs"),
        }
        per_class = result.get("per_class_map50") or None
        report_result(
            req_no=req_no,
            passed=passed,
            metrics=metrics,
            model_path=str(model_path) if model_path else None,
            notes=notes or f"base_trainer 自動回寫: {result.get('tier', '')}",
            per_class_metrics=per_class,
        )
    except Exception as e:
        print(f"[WARN] modelhub_report 回寫失敗（不中斷）: {e}", flush=True)
