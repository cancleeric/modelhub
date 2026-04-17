"""
modelhub_report.py — 訓練完成自動回寫 ModelHub DB（Sprint 8.2）

訓練腳本尾部呼叫 report_result()，透過 ModelHub REST API 更新 submission 狀態。
API 不可達時只印 warning，不中斷訓練主流程。

環境變數：
  MODELHUB_API_URL   — 預設 http://localhost:8000
  MODELHUB_API_KEY   — 預設 modelhub-dev-key-2026（bootstrap key）
"""

import os
import sys

# Python 版本相容（3.9 沒有 str | None union type hint at runtime for older versions）
from typing import Optional


def report_result(
    req_no: str,
    passed: bool,
    metrics: dict,
    model_path: Optional[str] = None,
    notes: str = "",
) -> bool:
    """
    訓練完成後呼叫此函式，自動回寫 ModelHub submission status。

    Args:
        req_no:     工單編號，例如 "MH-2026-006"
        passed:     True = trained（pass/baseline），False = training_failed
        metrics:    dict，例如 {"map50": 0.62, "epochs": 20}
        model_path: 最佳模型檔案路徑（可選）
        notes:      附加備註（可選）

    Returns:
        True  = API 更新成功
        False = 失敗（只 print warning，不 raise）
    """
    try:
        import requests
    except ImportError:
        print("[modelhub_report] WARN: requests 套件未安裝，跳過回寫", file=sys.stderr)
        return False

    base = os.environ.get("MODELHUB_API_URL", "http://localhost:8000")
    key = os.environ.get("MODELHUB_API_KEY", "modelhub-dev-key-2026")

    status = "trained" if passed else "training_failed"
    payload = {
        "status": status,
        "metrics": metrics,
        "model_path": model_path,
        "notes": notes,
    }

    try:
        resp = requests.patch(
            f"{base}/api/submissions/{req_no}/training-result",
            json=payload,
            headers={"X-Api-Key": key},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"[modelhub_report] {req_no} -> {status} (HTTP {resp.status_code})", flush=True)
        return True
    except Exception as e:
        print(f"[modelhub_report] WARN: 回寫失敗，不中斷訓練: {e}", file=sys.stderr, flush=True)
        return False
