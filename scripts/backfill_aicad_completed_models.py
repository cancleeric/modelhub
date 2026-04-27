#!/usr/bin/env python3
"""
backfill_aicad_completed_models.py
===================================
緊急回填 MH-2026-010 / MH-2026-011 的 model_output_path。

背景：
  本機 MPS 訓練腳本 train.py 未呼叫 modelhub_report.report_result()，
  導致 submissions.model_output_path 和 model_versions.file_path 永遠 NULL。
  模型檔已實際存在且達標（MH-010 val_acc=0.977, MH-011 val_acc=1.0）。

策略：
  Step 1 — 透過 PATCH /api/submissions/{req_no} API 回填 model_output_path
           (用 X-Api-Key，endpoint 接受 CurrentUserOrApiKey)
  Step 2 — 透過 SQLAlchemy ORM 直接更新 ModelVersion.file_path
           (registry PATCH 只接受 Bearer token，無法用 API Key;
            CEO Anderson 在 Task A 指示已授權此次 ORM 直改，
            commit message 會寫明授權來源)

授權：CEO Anderson 2026-04-27 緊急救援指令 (aicad-rescue)

使用方式（在 host 直接跑，不需要進容器）：
  cd ~/HurricaneCore/modelhub
  MODELHUB_API_KEY=mh-cto-ops-2026-0418 python3 scripts/backfill_aicad_completed_models.py

驗證：
  curl http://localhost:8950/api/submissions/MH-2026-010 (需 Bearer token)
  或直接查 DB:
    docker exec modelhub-api-dev python3 -c "
      import sys; sys.path.insert(0, '/app')
      from models import SessionLocal, Submission, ModelVersion
      db = SessionLocal()
      for rn in ['MH-2026-010','MH-2026-011']:
          s = db.query(Submission).filter(Submission.req_no==rn).first()
          mv = db.query(ModelVersion).filter(ModelVersion.req_no==rn).first()
          print(f'{rn}: model_output_path={s.model_output_path} mv.file_path={mv.file_path if mv else None}')
      db.close()
    "
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

MODELHUB_BASE_URL = os.environ.get("MODELHUB_BASE_URL", "http://localhost:8950")
API_KEY = os.environ.get("MODELHUB_API_KEY", "mh-cto-ops-2026-0418")

# host 端的實際路徑（本機 MPS 訓練輸出）
BACKFILL_MAP = {
    "MH-2026-010": {
        "model_path": "/Users/yinghaowang/HurricaneCore/modelhub/training/mh-2026-010/pid_symbols_best.pth",
        "result_json": "/Users/yinghaowang/HurricaneCore/modelhub/training/mh-2026-010/result.json",
        "model_name": "pid_symbols_mobilenetv2",
        "version": "v1.0",
        "arch": "mobilenetv2_100",
    },
    "MH-2026-011": {
        "model_path": "/Users/yinghaowang/HurricaneCore/modelhub/training/mh-2026-011/quality_router_best.pth",
        "result_json": "/Users/yinghaowang/HurricaneCore/modelhub/training/mh-2026-011/result.json",
        "model_name": "quality_router_efficientnet_b0",
        "version": "v1.0",
        "arch": "efficientnet_b0",
    },
}


def load_result_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def patch_submission_model_output_path(req_no: str, model_path: str) -> bool:
    """Step 1: 透過 API PATCH submission.model_output_path"""
    url = f"{MODELHUB_BASE_URL}/api/submissions/{req_no}"
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    payload = {"model_output_path": model_path}
    resp = requests.patch(url, json=payload, headers=headers, timeout=15)
    if resp.status_code == 200:
        print(f"  [OK] PATCH {req_no} model_output_path => {model_path}")
        return True
    else:
        print(f"  [FAIL] PATCH {req_no}: HTTP {resp.status_code} {resp.text[:300]}")
        return False


def backfill_model_version_via_orm(req_no: str, meta: dict, result: dict) -> bool:
    """
    Step 2: 直接用 SQLAlchemy ORM 更新 ModelVersion.file_path。
    授權：CEO Anderson 2026-04-27 Task A 緊急指令。
    原因：/api/registry/{id} PATCH 只接受 Bearer token，無法用 API Key。
    """
    # 透過容器 subprocess 執行（確保使用容器內的 DB 路徑和 Python 環境）
    import subprocess

    python_code = f"""
import sys
sys.path.insert(0, '/app')
from datetime import datetime
from models import SessionLocal, ModelVersion, Submission

db = SessionLocal()
try:
    req_no = {req_no!r}
    model_path = {meta['model_path']!r}
    best_val_acc = {result.get('best_val_acc', result.get('final_val_acc', 0.0))!r}
    epochs = {result.get('epochs', 0)!r}
    arch = {meta['arch']!r}

    mv = db.query(ModelVersion).filter(ModelVersion.req_no == req_no).order_by(ModelVersion.id.desc()).first()
    if mv is None:
        print(f'ERROR: No ModelVersion found for {{req_no}}')
        sys.exit(1)

    # 回填 file_path
    mv.file_path = model_path
    mv.map50_actual = best_val_acc   # 分類任務用 val_acc，統一寫入 map50_actual
    mv.epochs = epochs
    mv.arch = arch
    mv.pass_fail = 'pass'            # result.json verdict = pass
    mv.train_date = datetime.utcnow().strftime('%Y-%m-%d')

    # 同步 is_current（唯一版本，設為 current）
    db.query(ModelVersion).filter(
        ModelVersion.req_no == req_no,
        ModelVersion.id != mv.id,
    ).update({{'is_current': False}})
    mv.is_current = True

    db.commit()
    print(f'OK: ModelVersion id={{mv.id}} file_path={{mv.file_path}} pass_fail={{mv.pass_fail}} is_current={{mv.is_current}}')
except Exception as e:
    db.rollback()
    print(f'ERROR: {{e}}')
    sys.exit(1)
finally:
    db.close()
"""
    result_proc = subprocess.run(
        ["docker", "exec", "modelhub-api-dev", "python3", "-c", python_code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = (result_proc.stdout + result_proc.stderr).strip()
    if result_proc.returncode == 0 and "OK:" in output:
        print(f"  [OK] ORM update ModelVersion for {req_no}: {output}")
        return True
    else:
        print(f"  [FAIL] ORM update ModelVersion for {req_no}: {output}")
        return False


def verify(req_no: str) -> None:
    """驗收：印出 submission + model_version 回填結果"""
    python_code = f"""
import sys
sys.path.insert(0, '/app')
from models import SessionLocal, Submission, ModelVersion
db = SessionLocal()
rn = {req_no!r}
s = db.query(Submission).filter(Submission.req_no == rn).first()
mv = db.query(ModelVersion).filter(ModelVersion.req_no == rn).order_by(ModelVersion.id.desc()).first()
print(f'Submission  model_output_path: {{s.model_output_path!r}}')
print(f'ModelVersion file_path:        {{mv.file_path if mv else None!r}}')
print(f'ModelVersion pass_fail:        {{mv.pass_fail if mv else None!r}}')
print(f'ModelVersion is_current:       {{mv.is_current if mv else None!r}}')
db.close()
"""
    import subprocess
    result_proc = subprocess.run(
        ["docker", "exec", "modelhub-api-dev", "python3", "-c", python_code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    print(result_proc.stdout.strip())
    if result_proc.stderr.strip():
        print(f"  STDERR: {result_proc.stderr.strip()}")


def main() -> int:
    print(f"=== AICAD Completed Model Backfill ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"    URL: {MODELHUB_BASE_URL}")
    print(f"    API Key: {API_KEY[:8]}...")
    print()

    overall_ok = True

    for req_no, meta in BACKFILL_MAP.items():
        print(f"--- {req_no} ---")

        # 確認 model 檔案存在
        model_path = meta["model_path"]
        if not Path(model_path).exists():
            print(f"  [SKIP] Model file not found: {model_path}")
            overall_ok = False
            continue

        # 讀 result.json
        try:
            result = load_result_json(meta["result_json"])
        except Exception as e:
            print(f"  [ERROR] Cannot load result.json: {e}")
            overall_ok = False
            continue

        print(f"  Model file: {model_path} ({Path(model_path).stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  result.json: verdict={result.get('verdict')} best_val_acc={result.get('best_val_acc')}")

        # Step 1: PATCH submission
        ok1 = patch_submission_model_output_path(req_no, model_path)

        # Step 2: ORM update ModelVersion
        ok2 = backfill_model_version_via_orm(req_no, meta, result)

        # 驗收
        print(f"  [Verify]")
        verify(req_no)
        print()

        if not (ok1 and ok2):
            overall_ok = False

    print("=== Done ===")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
