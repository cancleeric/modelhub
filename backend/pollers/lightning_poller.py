"""
Lightning AI Job poller — Sprint 15 P2-1 / Sprint 17 P0-3

每 120 秒 poll 所有 platform=lightning 且 status=training 的 submission：
- 狀態變化寫 submission_history
- complete 時下載 output → 解析 metrics → 建 ModelVersion
- error 時標記 training_failed
- >48 小時未完成 → 通知 CTO overtime

需要：LIGHTNING_API_KEY env var。
如果 env var 未設，poller 還是會啟動但會 log warning，並跳過 API 呼叫。

Sprint 17 P0-3: 改為 APScheduler 模式（poll_once + start_scheduler），
與 kaggle_poller 架構一致，由 main.py lifespan 統一啟動。

TODO: 所有標記 TODO 的區塊待取得 LIGHTNING_API_KEY 後補完。
      取得方式：https://lightning.ai → Settings → API Keys
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from models import SessionLocal, Submission, ModelVersion
from parsers import parse_training_log
from notifications import notify_event

logger = logging.getLogger("modelhub.poller.lightning")

POLL_INTERVAL_SECONDS = int(os.environ.get("MODELHUB_LIGHTNING_POLL_INTERVAL", "120"))
OVERTIME_HOURS = int(os.environ.get("MODELHUB_TRAINING_OVERTIME_HOURS", "48"))
LIGHTNING_DOWNLOAD_DIR = os.environ.get("MODELHUB_LIGHTNING_DL_DIR", "/tmp/modelhub-lightning")

# Lightning AI 計費（T4 免費 22hr/月，超量 $0.5/hr 估算）
GPU_USD_PER_HOUR = float(os.environ.get("MODELHUB_LIGHTNING_GPU_USD_PER_HOUR", "0.5"))
LIGHTNING_IS_FREE_TIER = os.environ.get("LIGHTNING_IS_FREE_TIER", "true").lower() == "true"


def _lightning_env_ready() -> bool:
    return bool(os.environ.get("LIGHTNING_API_KEY"))


def _get_lightning_studio(studio_name: str):
    """
    取得 Lightning Studio 實例（lazy import）。
    TODO: 待 API Key 後實作。
    """
    try:
        from lightning_sdk import Studio  # type: ignore
        username = os.environ.get("LIGHTNING_USERNAME", "boardgamegroup")
        teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "boardgamegroup")
        studio = Studio(name=studio_name, teamspace=teamspace)
        return studio
    except Exception as exc:
        logger.warning("Lightning SDK unavailable: %s", exc)
        return None


def _fetch_job_status(studio_name: str) -> Optional[dict]:
    """
    查詢 Lightning Studio job 狀態（同步版本，供 APScheduler 呼叫）。

    TODO: 待 API Key 後補完。
    預期回傳格式：
        {"status": "running"/"complete"/"error"/"queued", "raw": str}
    """
    if not _lightning_env_ready():
        logger.warning("LIGHTNING_API_KEY 未設，跳過 status fetch")
        return None

    # TODO: 實作
    # studio = _get_lightning_studio(studio_name)
    # if not studio:
    #     return None
    # status = studio.status  # "running" / "stopped" / "error" 等
    # return {"status": _normalize_status(status), "raw": str(status)}
    logger.debug("[TODO] _fetch_job_status studio=%s（待 API Key 實作）", studio_name)
    return None


def _download_job_output(studio_name: str, dest_dir: str) -> Optional[str]:
    """
    下載 Lightning job output（result.json + best.pt 等）。

    TODO: 待 API Key 後補完。
    Lightning Studio 產出會在 /teamspace/studios/this_studio/kaggle/working/ 等路徑。
    """
    if not _lightning_env_ready():
        return None

    # TODO: 實作
    # studio = _get_lightning_studio(studio_name)
    # studio.download("result.json", dest_dir)
    logger.debug("[TODO] _download_job_output studio=%s（待 API Key 實作）", studio_name)
    return None


def _normalize_status(raw_status: str) -> str:
    """Lightning status 正規化為 modelhub 標準狀態"""
    mapping = {
        "running": "running",
        "stopped": "complete",
        "failed": "error",
        "error": "error",
        "queued": "queued",
        "pending": "queued",
        "starting": "queued",
    }
    return mapping.get(raw_status.lower(), raw_status.lower())


def _append_history(db: Session, req_no: str, action: str, meta: Optional[dict] = None,
                    note: Optional[str] = None) -> None:
    from models import SubmissionHistory
    row = SubmissionHistory(
        req_no=req_no,
        action=action,
        actor="lightning-poller",
        note=note,
        meta=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(row)


def _process_submission(
    db: Session,
    submission: "Submission",
) -> None:
    """
    處理單一 Lightning submission 的 poll 週期（同步版本）。
    結構與 kaggle_poller._process_submission 一致。

    Sprint 17 P1-4: 改讀 submission.lightning_studio_name（若有設定）。
    TODO: 完整邏輯待 API Key 後補完。
    """
    # P1-4: 優先讀 lightning_studio_name 欄位
    studio_name = getattr(submission, "lightning_studio_name", None) or \
                  getattr(submission, "kaggle_kernel_slug", None)
    if not studio_name:
        logger.warning("submission %s 無 studio_name，跳過", submission.req_no)
        return

    status_info = _fetch_job_status(studio_name)
    if status_info is None:
        # API Key 未設或 TODO 尚未實作，靜默跳過
        return

    status = status_info.get("status", "unknown")
    logger.info("submission %s studio=%s status=%s", submission.req_no, studio_name, status)

    # 狀態變化寫 submission_history
    if submission.kaggle_status != status:
        submission.kaggle_status = status
        submission.kaggle_status_updated_at = datetime.utcnow()
        _append_history(db, req_no=submission.req_no, action="lightning_status_change",
                        meta={"new_status": status})
        db.commit()

    if status == "complete":
        # TODO: 下載 output → 解析 metrics → 建 ModelVersion
        dest_dir = str(Path(LIGHTNING_DOWNLOAD_DIR) / str(submission.req_no))
        output_dir = _download_job_output(studio_name, dest_dir)
        if output_dir:
            result_json_candidates = list(Path(output_dir).rglob("result.json"))
            if result_json_candidates:
                raw_log = result_json_candidates[0].read_text()
                metrics = parse_training_log(raw_log)
                logger.info("submission %s metrics=%s", submission.req_no, metrics)
                # TODO: create ModelVersion（參考 kaggle_poller 實作）
            else:
                logger.warning("submission %s 無 result.json，無法解析 metrics", submission.req_no)

    elif status == "error":
        submission.status = "failed"
        db.commit()
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(notify_event("training_failed", submission))
            else:
                loop.run_until_complete(notify_event("training_failed", submission))
        except Exception:
            pass

    # Overtime 偵測
    training_started = getattr(submission, "training_started_at", None)
    if training_started and status in ("running", "queued"):
        elapsed = datetime.utcnow() - training_started
        if elapsed > timedelta(hours=OVERTIME_HOURS):
            logger.warning(
                "submission %s overtime! studio=%s elapsed=%.1fh",
                submission.req_no, studio_name, elapsed.total_seconds() / 3600,
            )
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(notify_event("training_overtime", submission))
                else:
                    loop.run_until_complete(notify_event("training_overtime", submission))
            except Exception:
                pass


def poll_once() -> dict:
    """
    掃一輪 platform=lightning 且 status=training 的 submission（同步，供 APScheduler 呼叫）。
    LIGHTNING_API_KEY 未設時直接 return。
    """
    global _last_poll_at
    _last_poll_at = datetime.utcnow()

    if not _lightning_env_ready():
        logger.debug("LIGHTNING_API_KEY not set, skip lightning poll")
        return {"skipped": True, "reason": "LIGHTNING_API_KEY not set"}

    db: Session = SessionLocal()
    try:
        # TODO: 實際查詢 platform=lightning 的 submission
        # rows = (
        #     db.query(Submission)
        #     .filter(
        #         Submission.platform == "lightning",
        #         Submission.status.in_(["training", "queued", "running"]),
        #     )
        #     .all()
        # )
        # summary = {"checked": len(rows), "changed": 0}
        # for sub in rows:
        #     _process_submission(db, sub)
        logger.debug("[TODO] Lightning poller tick（待 API Key 後補 query）")
        return {"skipped": True, "reason": "not implemented"}
    except Exception as exc:
        logger.exception("Lightning poller poll_once 例外: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Scheduler lifecycle（main.py 呼叫）
# ---------------------------------------------------------------------------

_scheduler = None
_last_poll_at: Optional[datetime] = None


def get_last_poll_at() -> Optional[datetime]:
    return _last_poll_at


def start_scheduler():
    global _scheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        logger.warning("apscheduler not installed, lightning poller disabled")
        return None

    if _scheduler:
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone="Asia/Taipei")
    _scheduler.add_job(
        poll_once, "interval",
        seconds=POLL_INTERVAL_SECONDS,
        id="lightning-poller",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info("Lightning poller scheduler started (interval=%ds)", POLL_INTERVAL_SECONDS)
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Lightning scheduler stopped")
