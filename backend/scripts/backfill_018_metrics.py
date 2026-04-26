"""
backfill_018_metrics.py — 補寫 MH-2026-018 PPE v1 metrics

Root cause 已確認：
  read_log_files 的 5MB 上限將 PPE NDJSON log（5.5MB）截斷為空字串，
  parser 收到空字串 → metrics={} → model_versions map50/map50_95/pass_fail 全 None。

此 script 從已下載的 PPE log（/tmp/k018-output/）重新解析 metrics，
透過 SQLAlchemy ORM 更新 model_versions（id=9），不直接執行 SQL UPDATE。

執行方式：
    cd /Users/yinghaowang/HurricaneCore/modelhub/backend
    DATABASE_URL=sqlite:////Users/yinghaowang/HurricaneCore/docker-data/modelhub/db/modelhub.db \
        python3 scripts/backfill_018_metrics.py

乾跑（只印不寫）：
    ... python3 scripts/backfill_018_metrics.py --dry-run
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

# 加入 backend 路徑
_BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backfill-018")

REQ_NO = "MH-2026-018"
LOG_DIR = "/tmp/k018-output"
MAP50_THRESHOLD = 0.7  # 從 submissions.map50_threshold 取得的值


def parse_args():
    p = argparse.ArgumentParser(description="Backfill MH-2026-018 PPE v1 metrics")
    p.add_argument("--dry-run", action="store_true", help="只印不寫 DB")
    return p.parse_args()


def extract_metrics_from_log(log_dir: str) -> dict:
    """從 PPE log 解析 mAP50 / mAP50-95 / per_class。"""
    from utils import read_log_files
    from parsers.yolo import parse_yolo_log

    log_text = read_log_files(log_dir)
    if not log_text:
        raise RuntimeError(f"read_log_files 回傳空字串，請確認 {log_dir} 存在且含 .log 檔")

    logger.info("log_text length: %d bytes", len(log_text))
    result = parse_yolo_log(log_text)
    logger.info("parse_yolo_log result: %s", result)
    return result


def backfill(dry_run: bool) -> None:
    from models import SessionLocal, ModelVersion, Submission

    parsed = extract_metrics_from_log(LOG_DIR)
    metrics = parsed.get("metrics", {})
    per_class = parsed.get("per_class") or {}

    map50 = metrics.get("map50")
    map50_95 = metrics.get("map50_95")

    if map50 is None:
        raise RuntimeError(f"解析失敗：map50=None，metrics={metrics}")

    pass_fail = "pass" if map50 >= MAP50_THRESHOLD else "fail"

    logger.info(
        "Parsed: map50=%.4f map50_95=%s pass_fail=%s threshold=%.2f per_class=%s",
        map50, f"{map50_95:.4f}" if map50_95 is not None else "None",
        pass_fail, MAP50_THRESHOLD, per_class,
    )

    if dry_run:
        logger.info("[DRY RUN] 不寫 DB")
        print(json.dumps({
            "req_no": REQ_NO,
            "map50": map50,
            "map50_95": map50_95,
            "pass_fail": pass_fail,
            "per_class": per_class,
            "dry_run": True,
        }, indent=2))
        return

    db = SessionLocal()
    try:
        # 找 model_versions（REQ_NO + version=v1）
        mv = (
            db.query(ModelVersion)
            .filter(ModelVersion.req_no == REQ_NO, ModelVersion.version == "v1")
            .first()
        )
        if not mv:
            raise RuntimeError(f"找不到 model_versions: req_no={REQ_NO} version=v1")

        logger.info("Before update: id=%d map50=%s map50_95=%s pass_fail=%s",
                    mv.id, mv.map50, mv.map50_95, mv.pass_fail)

        mv.map50 = map50
        mv.map50_95 = map50_95
        mv.map50_actual = map50
        mv.map50_95_actual = map50_95
        mv.pass_fail = pass_fail

        # 同步更新 submission.per_class_metrics（若有 per_class 資料）
        if per_class:
            sub = db.query(Submission).filter(Submission.req_no == REQ_NO).first()
            if sub:
                sub.per_class_metrics = json.dumps(per_class, ensure_ascii=False)
                logger.info("Updated submission.per_class_metrics")

        db.commit()
        db.refresh(mv)

        logger.info("After update: id=%d map50=%.4f map50_95=%s pass_fail=%s",
                    mv.id, mv.map50,
                    f"{mv.map50_95:.4f}" if mv.map50_95 is not None else "None",
                    mv.pass_fail)

        print(json.dumps({
            "req_no": REQ_NO,
            "model_version_id": mv.id,
            "map50": mv.map50,
            "map50_95": mv.map50_95,
            "pass_fail": mv.pass_fail,
            "per_class": per_class,
            "status": "updated",
        }, indent=2))

    finally:
        db.close()


if __name__ == "__main__":
    args = parse_args()
    backfill(dry_run=args.dry_run)
