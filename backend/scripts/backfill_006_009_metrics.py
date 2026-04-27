"""
backfill_006_009_metrics.py — 補寫 MH-2026-006 / MH-2026-009 的 model_versions metrics

Root cause（已確認 2026-04-27）：
  Kaggle kernel output 目錄未輸出 result.json（僅含 dataset / model weights），
  導致 _read_result_json() 回傳 {}，log parser 亦無法解析 → metrics 全 None。

修復策略：
  MH-006（YOLO 文字偵測 v3）：
    - 從 submissions.per_class_metrics {"text": 0.4489} 計算 map50 均值 = 0.4489
    - map50_threshold = 0.5974，verdict = fail
    - 注意：v2（id=5）是本機 resume 的最佳版本（map50=0.5993），不動；
      本腳本只修 v3（id=12）—— poller auto-filled 但 map50 為 None 的版本

  MH-009（TrOCR OCR v1）：
    - 從 training/mh-2026-009/result.json 讀 CER / exact_match
    - exact_match = 0.036 → map50（OCR 任務以 exact_match 作主指標）
    - map50_threshold = None（OCR 任務無 mAP 門檻），pass_fail = None
    - ocr_cer = 0.4227 寫入 notes

執行方式（在 host 直接跑，使用 host 路徑的 DB）：
    cd /Users/yinghaowang/HurricaneCore/modelhub/backend
    DATABASE_URL=sqlite:////Users/yinghaowang/HurricaneCore/docker-data/modelhub/db/modelhub.db \\
        python3 scripts/backfill_006_009_metrics.py

乾跑（只印不寫）：
    ... python3 scripts/backfill_006_009_metrics.py --dry-run

授權：CEO Anderson 2026-04-27 Task A 回填授權（mh-rescue 工單）
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

_BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backfill-006-009")

# training/ 目錄：優先 env var 覆蓋，再找本機路徑
_TRAINING_CANDIDATES = [
    Path(os.environ.get("MODELHUB_TRAINING_DIR", "")),
    Path("/Users/yinghaowang/HurricaneCore/modelhub/training"),
    Path("/app/training"),
]
TRAINING_BASE = next((p for p in _TRAINING_CANDIDATES if p and p.exists()), _TRAINING_CANDIDATES[1])

# MH-009 result.json 備用路徑（腳本目錄下，供容器 exec 使用）
_SCRIPT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# MH-006 設定
# ---------------------------------------------------------------------------
MH006_REQ_NO = "MH-2026-006"
MH006_VERSION = "v3"          # poller auto-filled，map50=None 的版本
MH006_THRESHOLD = 0.5974      # submissions.map50_threshold

# per_class_metrics 已寫在 submissions（{"text": 0.4489}）
# map50 = mean of per_class = 0.4489
MH006_MAP50 = 0.4489
MH006_MAP50_95 = None         # Kaggle kernel 未輸出 map50_95


# ---------------------------------------------------------------------------
# MH-009 設定
# ---------------------------------------------------------------------------
MH009_REQ_NO = "MH-2026-009"
MH009_VERSION = "v1"
# 找 result.json：先查 training dir，再查 scripts/ 旁的備用檔
_MH009_CANDIDATES = [
    TRAINING_BASE / "mh-2026-009" / "result.json",
    _SCRIPT_DIR / "mh009_result_tmp.json",
]
MH009_RESULT_JSON = next((p for p in _MH009_CANDIDATES if p.exists()), _MH009_CANDIDATES[0])
MH009_THRESHOLD = None        # OCR 任務無 mAP50 門檻


def _load_mh009_metrics() -> dict:
    """從本機 result.json 讀取 MH-009 OCR 指標。"""
    if not MH009_RESULT_JSON.exists():
        raise FileNotFoundError(f"找不到 MH-009 result.json: {MH009_RESULT_JSON}")
    obj = json.loads(MH009_RESULT_JSON.read_text())
    exact_match = obj.get("test_exact_match") or obj.get("exact_match")
    cer = obj.get("test_cer") or obj.get("val_cer") or obj.get("cer")
    epochs = obj.get("epochs")
    if exact_match is None:
        raise ValueError(f"result.json 缺少 exact_match 欄位: {obj}")
    return {
        "map50": float(exact_match),       # exact_match 作為 OCR 主要品質指標
        "ocr_cer": float(cer) if cer else None,
        "epochs": int(epochs) if epochs else None,
    }


def backfill_mh006(db, dry_run: bool) -> dict:
    """回填 MH-006 v3 model_version。"""
    from models import ModelVersion

    map50 = MH006_MAP50
    pass_fail = "fail" if map50 < MH006_THRESHOLD else "pass"

    logger.info(
        "[MH-006] map50=%.4f threshold=%.4f → pass_fail=%s (version=%s)",
        map50, MH006_THRESHOLD, pass_fail, MH006_VERSION,
    )

    if dry_run:
        result = {
            "req_no": MH006_REQ_NO,
            "version": MH006_VERSION,
            "map50": map50,
            "map50_95": MH006_MAP50_95,
            "pass_fail": pass_fail,
            "dry_run": True,
        }
        print(json.dumps(result, indent=2))
        return result

    mv = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.req_no == MH006_REQ_NO,
            ModelVersion.version == MH006_VERSION,
        )
        .first()
    )
    if not mv:
        raise RuntimeError(f"找不到 model_versions: req_no={MH006_REQ_NO} version={MH006_VERSION}")

    logger.info("[MH-006] Before: id=%d map50=%s pass_fail=%s", mv.id, mv.map50, mv.pass_fail)

    mv.map50 = map50
    mv.map50_actual = map50
    mv.map50_95 = MH006_MAP50_95
    mv.map50_95_actual = MH006_MAP50_95
    mv.pass_fail = pass_fail
    existing_notes = mv.notes or ""
    if "per_class_metrics fallback" not in existing_notes:
        mv.notes = (
            existing_notes.rstrip(";").strip()
            + f"; backfill 2026-04-27 from per_class_metrics {{text: 0.4489}}; CER=N/A (detection task)"
        )

    logger.info("[MH-006] After: id=%d map50=%.4f pass_fail=%s", mv.id, mv.map50, mv.pass_fail)
    return {"req_no": MH006_REQ_NO, "model_version_id": mv.id, "map50": mv.map50, "pass_fail": mv.pass_fail}


def backfill_mh009(db, dry_run: bool) -> dict:
    """回填 MH-009 v1 model_version（OCR 任務）。"""
    from models import ModelVersion

    m = _load_mh009_metrics()
    map50 = m["map50"]
    ocr_cer = m["ocr_cer"]
    epochs = m["epochs"]

    # OCR 任務：map50_threshold=None，不做 pass/fail
    pass_fail = None

    notes_parts = [f"backfill 2026-04-27 from result.json"]
    notes_parts.append(f"exact_match={map50:.4f}")
    if ocr_cer is not None:
        notes_parts.append(f"CER={ocr_cer:.4f}")
    notes_parts.append("OCR 任務以 exact_match 作主指標（無 mAP50 門檻）")

    logger.info(
        "[MH-009] exact_match(map50)=%.4f ocr_cer=%s epochs=%s → pass_fail=%s (version=%s)",
        map50, ocr_cer, epochs, pass_fail, MH009_VERSION,
    )

    if dry_run:
        result = {
            "req_no": MH009_REQ_NO,
            "version": MH009_VERSION,
            "map50": map50,
            "ocr_cer": ocr_cer,
            "pass_fail": pass_fail,
            "epochs": epochs,
            "dry_run": True,
        }
        print(json.dumps(result, indent=2))
        return result

    mv = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.req_no == MH009_REQ_NO,
            ModelVersion.version == MH009_VERSION,
        )
        .first()
    )
    if not mv:
        raise RuntimeError(f"找不到 model_versions: req_no={MH009_REQ_NO} version={MH009_VERSION}")

    logger.info("[MH-009] Before: id=%d map50=%s pass_fail=%s epochs=%s", mv.id, mv.map50, mv.pass_fail, mv.epochs)

    mv.map50 = map50
    mv.map50_actual = map50
    mv.pass_fail = pass_fail
    if epochs is not None:
        mv.epochs = epochs
    mv.notes = "; ".join(notes_parts)

    logger.info("[MH-009] After: id=%d map50=%.4f pass_fail=%s epochs=%s", mv.id, mv.map50, mv.pass_fail, mv.epochs)
    return {
        "req_no": MH009_REQ_NO,
        "model_version_id": mv.id,
        "map50": mv.map50,
        "ocr_cer": ocr_cer,
        "pass_fail": mv.pass_fail,
    }


def verify(db) -> None:
    """驗收：印出 model_versions 回填結果。"""
    from models import ModelVersion
    print("\n=== Verification ===")
    for req_no, version in [(MH006_REQ_NO, MH006_VERSION), (MH009_REQ_NO, MH009_VERSION)]:
        mv = (
            db.query(ModelVersion)
            .filter(ModelVersion.req_no == req_no, ModelVersion.version == version)
            .first()
        )
        if mv:
            print(f"  {req_no} {version}: map50={mv.map50} map50_95={mv.map50_95} pass_fail={mv.pass_fail} notes={mv.notes[:80] if mv.notes else None}")
        else:
            print(f"  {req_no} {version}: NOT FOUND")


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill MH-006/MH-009 metrics into model_versions")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"=== Backfill MH-006 + MH-009 Metrics (dry_run={args.dry_run}) ===")

    from models import SessionLocal

    db = SessionLocal()
    overall_ok = True
    try:
        # MH-006
        print(f"\n--- {MH006_REQ_NO} ---")
        try:
            backfill_mh006(db, args.dry_run)
        except Exception as e:
            logger.error("[MH-006] FAILED: %s", e)
            overall_ok = False

        # MH-009
        print(f"\n--- {MH009_REQ_NO} ---")
        try:
            backfill_mh009(db, args.dry_run)
        except Exception as e:
            logger.error("[MH-009] FAILED: %s", e)
            overall_ok = False

        if not args.dry_run:
            db.commit()
            verify(db)
    except Exception as e:
        db.rollback()
        logger.error("Fatal: %s", e)
        return 1
    finally:
        db.close()

    print("\n=== Done ===")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
