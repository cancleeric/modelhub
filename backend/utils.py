"""
utils.py — ModelHub 共用工具函式

P1-8: 抽出各 poller 共用的 _next_version_for()，避免三份重複實作。
P3-28: 抽出各 poller 共用的 read_log_files()，避免 kaggle_poller / ssh_poller 重複定義。
"""

import re
from pathlib import Path
from sqlalchemy.orm import Session


def read_log_files(dir_path: str, max_size: int = 20 * 1024 * 1024) -> str:
    """
    P3-28: 讀取目錄下所有 log/txt/json/out/stdout 檔案，concat 回傳（最大 max_size bytes）。
    原本各 poller 各有一份實作，統一抽到這裡。

    檔案讀取優先序：
    1. .log / .out / .stdout（training log，最重要，優先讀）
    2. .json（result.json 等，次之）
    3. .txt（輔助資料）
    並且 .log 採用單獨 20MB 限額，避免大型 NDJSON log（如 Kaggle PPE/YOLOv8m
    輸出可達 5–10MB）被 5MB 硬限截斷而導致 metrics 解析失敗。
    """
    root = Path(dir_path)
    if not root.exists():
        return ""

    # 分兩批：log 優先，其他次之
    _PRIORITY_EXTS = {".log", ".out", ".stdout"}
    _SECONDARY_EXTS = {".json", ".txt"}

    def _collect(exts: set, size_limit: int) -> list[str]:
        parts: list[str] = []
        used = 0
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in exts:
                continue
            try:
                size = f.stat().st_size
                if used + size > size_limit:
                    continue  # skip（不 break）— 嘗試更小的下一個檔案
                parts.append(f.read_text(errors="replace"))
                used += size
            except Exception:
                continue
        return parts

    acc = _collect(_PRIORITY_EXTS, max_size)
    acc += _collect(_SECONDARY_EXTS, max_size)
    return "\n".join(acc)


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
