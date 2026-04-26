"""
utils.py — ModelHub 共用工具函式

P1-8: 抽出各 poller 共用的 _next_version_for()，避免三份重複實作。
P3-28: 抽出各 poller 共用的 read_log_files()，避免 kaggle_poller / ssh_poller 重複定義。
M16: 抽出 extract_epoch_curve()，讓三個 poller 都能用同一個邏輯寫入 epoch_curve。
"""

import json
import re
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session


def read_log_files(dir_path: str, max_size: int = 5 * 1024 * 1024) -> str:
    """
    P3-28: 讀取目錄下所有 log/txt/json/out/stdout 檔案，concat 回傳（最大 max_size bytes）。
    原本各 poller 各有一份實作，統一抽到這裡。
    """
    acc = []
    total = 0
    root = Path(dir_path)
    if not root.exists():
        return ""
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".log", ".txt", ".json", ".out", ".stdout"):
            continue
        try:
            size = f.stat().st_size
            if total + size > max_size:
                break
            acc.append(f.read_text(errors="replace"))
            total += size
        except Exception:
            continue
    return "\n".join(acc)


def extract_epoch_curve(dest_dir: str, log_text: Optional[str] = None) -> list[dict]:
    """
    M16: 從 training output 目錄提取逐 epoch 訓練曲線資料。

    優先序：
    1. results.csv（YOLOv8 標準輸出，每 epoch 一行）
    2. parsed_result["epoch_data"]（由 parse_yolo_log 從 stdout 解析，作為 fallback）

    Args:
        dest_dir:  已下載的 kernel output 目錄路徑
        log_text:  可選，已讀取的 log 全文（若已讀過就傳入，避免重複讀）

    Returns:
        epoch_data list（可能為空 list）
    """
    from parsers.yolo import parse_results_csv

    root = Path(dest_dir)
    if not root.exists():
        return []

    # 1. 找 results.csv（YOLOv8 在 run output 目錄內）
    # 路徑例：<dest_dir>/mh-2026-019/yolo_run/results.csv 或 <dest_dir>/results.csv
    csv_candidates = sorted(root.rglob("results.csv"))
    for csv_path in csv_candidates:
        try:
            csv_text = csv_path.read_text(errors="replace")
            epoch_data = parse_results_csv(csv_text)
            if epoch_data:
                return epoch_data
        except Exception:
            continue

    # 2. Fallback：從已解析的 log_text 中取 epoch_data（由 parse_yolo_log 回傳）
    if log_text:
        try:
            from parsers import parse_training_log
            parsed = parse_training_log("yolov8m", log_text)
            epoch_data = parsed.get("epoch_data", [])
            if epoch_data:
                return epoch_data
        except Exception:
            pass

    return []


def next_version_for(db: Session, req_no: str) -> str:
    """
    查詢 req_no 目前最大 ModelVersion，回傳下一個版號字串（v1, v2, ...）。

    Args:
        db:     SQLAlchemy Session
        req_no: 需求單編號

    Returns:
        "v1" 若無任何既有版本，否則 "v{n+1}"
    """
    from models import ModelVersion
    latest = (
        db.query(ModelVersion)
        .filter(ModelVersion.req_no == req_no)
        .order_by(ModelVersion.id.desc())
        .first()
    )
    if not latest:
        return "v1"
    m = re.match(r"v(\d+)", latest.version or "")
    n = int(m.group(1)) + 1 if m else 1
    return f"v{n}"
