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
    直接呼叫 LightningLauncher.get_job_status() 取得正規化狀態。

    預期回傳格式：
        {"status": "running"/"complete"/"error"/"queued", "raw": str}
    """
    if not _lightning_env_ready():
        logger.warning("LIGHTNING_API_KEY 未設，跳過 status fetch")
        return None

    try:
        from resources.lightning_launcher import LightningLauncher
        launcher = LightningLauncher()
        raw_status = launcher.get_job_status(studio_name)
        normalized = _normalize_status(raw_status)
        logger.debug("_fetch_job_status studio=%s raw=%s normalized=%s", studio_name, raw_status, normalized)
        return {"status": normalized, "raw": raw_status}
    except Exception as exc:
        logger.warning("_fetch_job_status studio=%s exception: %s", studio_name, exc)
        return None


def _download_job_output(studio_name: str, dest_dir: str) -> Optional[str]:
    """
    下載 Lightning job output（best.pt + training log）。
    呼叫 LightningLauncher.download_model() 取得 best.pt，
    並嘗試取 training log 用於 metrics 解析。
    回傳 dest_dir 若成功，None 若失敗。
    """
    if not _lightning_env_ready():
        return None

    try:
        from pathlib import Path as _Path
        from resources.lightning_launcher import LightningLauncher
        launcher = LightningLauncher()
        dest = _Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        # 下載 best.pt
        best_pt_path = str(dest / "best.pt")
        pt_ok = launcher.download_model(studio_name, best_pt_path)
        if pt_ok:
            logger.info("_download_job_output: best.pt downloaded to %s", best_pt_path)

        # 嘗試取 training log（供 metrics 解析）
        log_text = launcher.get_job_logs(studio_name)
        if log_text:
            log_path = dest / "training.log"
            log_path.write_text(log_text, encoding="utf-8")
            logger.info("_download_job_output: training log written to %s (%d chars)", log_path, len(log_text))

        # 只要 best.pt 或 log 其中一個有拿到，視為下載成功
        if pt_ok or log_text:
            return dest_dir

        logger.warning("_download_job_output studio=%s: 未取得任何產出", studio_name)
        return None
    except Exception as exc:
        logger.warning("_download_job_output studio=%s exception: %s", studio_name, exc)
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


def _append_training_failed_summary_lightning(db: Session, sub: "Submission") -> None:
    """
    Sprint 19 C.1: 解析已下載的 Lightning output（若有），寫入含 partial metrics 的
    training_failed_summary history，供前端顯示上次失敗的 mAP50。
    """
    parsed_map50 = None
    completed_epochs = None
    try:
        dest_dir = str(Path(LIGHTNING_DOWNLOAD_DIR) / str(sub.req_no))
        from pollers.kaggle_poller import _read_log_files
        log_text = _read_log_files(dest_dir)
        if log_text:
            parsed = parse_training_log(sub.arch or "yolov8m", log_text)
            metrics = parsed.get("metrics", {}) or {}
            parsed_map50 = metrics.get("map50")
            completed_epochs = metrics.get("epochs")
    except Exception as e:
        logger.debug("_append_training_failed_summary_lightning parse failed: %s", e)

    _append_history(
        db,
        req_no=sub.req_no,
        action="training_failed_summary",
        meta={
            "partial_map50": parsed_map50,
            "epochs": completed_epochs,
            "verdict": "fail",
        },
        note=f"mAP50={parsed_map50 or 'N/A'}",
    )


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
        dest_dir = str(Path(LIGHTNING_DOWNLOAD_DIR) / str(submission.req_no))
        output_dir = _download_job_output(studio_name, dest_dir)
        metrics: dict = {}
        parsed: dict = {}
        if output_dir:
            # 讀取所有 log 文字（.log / .txt / .json）
            from pollers.kaggle_poller import _read_log_files
            log_text = _read_log_files(output_dir)
            if log_text:
                parsed = parse_training_log(submission.arch or "yolov8m", log_text)
                metrics = parsed.get("metrics", {}) or {}
                logger.info("submission %s metrics=%s", submission.req_no, metrics)

        now = datetime.utcnow()
        submission.status = "trained"
        submission.training_completed_at = now
        submission.kaggle_status = "complete"
        submission.kaggle_status_updated_at = now

        # GPU 時數估算
        if submission.training_started_at:
            gpu_seconds = int((now - submission.training_started_at).total_seconds())
            submission.gpu_seconds = gpu_seconds
            if LIGHTNING_IS_FREE_TIER:
                submission.estimated_cost_usd = 0.0
            else:
                submission.estimated_cost_usd = round(gpu_seconds / 3600 * GPU_USD_PER_HOUR, 4)

        # per-class metrics
        per_class = parsed.get("per_class") if parsed else None
        if per_class and isinstance(per_class, dict):
            import json as _json_pc
            try:
                submission.per_class_metrics = _json_pc.dumps(per_class, ensure_ascii=False)
            except Exception:
                pass

        # 建 ModelVersion
        from models import ModelVersion as _ModelVersion
        import re as _re

        def _next_version(req_no: str, db: Session) -> str:
            latest = (
                db.query(_ModelVersion)
                .filter(_ModelVersion.req_no == req_no)
                .order_by(_ModelVersion.id.desc())
                .first()
            )
            if not latest:
                return "v1"
            m2 = _re.match(r"v(\d+)", latest.version or "")
            n = int(m2.group(1)) + 1 if m2 else 1
            return f"v{n}"

        next_ver = _next_version(submission.req_no, db)
        mv = _ModelVersion(
            req_no=submission.req_no,
            product=submission.product,
            model_name=submission.req_name or f"{submission.product} model",
            version=next_ver,
            train_date=now.strftime("%Y-%m-%d"),
            map50=metrics.get("map50"),
            map50_95=metrics.get("map50_95"),
            map50_actual=metrics.get("map50"),
            map50_95_actual=metrics.get("map50_95"),
            epochs=metrics.get("epochs"),
            batch_size=metrics.get("batch_size"),
            arch=submission.arch,
            status="pending_acceptance",
            notes=f"auto-filled by lightning-poller at {now.isoformat()}",
        )
        db.add(mv)
        _append_history(
            db,
            req_no=submission.req_no,
            action="training_complete",
            meta={
                "metrics": metrics,
                "gpu_seconds": submission.gpu_seconds,
                "estimated_cost_usd": submission.estimated_cost_usd,
                "version": next_ver,
                "platform": "lightning",
            },
        )
        db.commit()
        db.refresh(submission)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(notify_event("training_complete", submission))
            else:
                loop.run_until_complete(notify_event("training_complete", submission))
        except Exception:
            pass

    elif status == "error":
        submission.status = "failed"

        # Sprint 19 C.1: 寫入失敗摘要 history（供前端顯示上次失敗的 mAP50）
        _append_training_failed_summary_lightning(db, submission)

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


def _check_lightning_quota_warning(db: Session) -> None:
    """
    Sprint 22 Task 22-4: Lightning 配額預警。
    本月剩餘 < 3h 且過去 1 小時沒發過時，寫 history + notify CTO。
    """
    QUOTA_WARN_THRESHOLD = float(os.environ.get("LIGHTNING_QUOTA_WARN_HOURS", "3"))
    try:
        from resources.prober import LightningQuotaTracker
        tracker = LightningQuotaTracker()
        remaining = tracker.get_remaining_hours(db)
        if remaining >= QUOTA_WARN_THRESHOLD:
            return

        # 1 小時節流
        from datetime import timedelta as _td
        cutoff = datetime.utcnow() - _td(hours=1)
        already = (
            db.query(Submission)
            .join(
                __import__("models", fromlist=["SubmissionHistory"]).SubmissionHistory,
                __import__("models", fromlist=["SubmissionHistory"]).SubmissionHistory.req_no == "__quota_lightning__",
            )
            .first()
        )
        # 用 SubmissionHistory 做節流記錄（簡化版：直接查）
        from models import SubmissionHistory
        already_sent = (
            db.query(SubmissionHistory)
            .filter(
                SubmissionHistory.req_no == "__quota_lightning__",
                SubmissionHistory.action == "lightning_quota_warning",
                SubmissionHistory.created_at >= cutoff,
            )
            .first()
        )
        if already_sent:
            return

        row = SubmissionHistory(
            req_no="__quota_lightning__",
            action="lightning_quota_warning",
            actor="lightning-poller",
            note=f"本月剩餘配額 {remaining:.1f}h（低於 {QUOTA_WARN_THRESHOLD}h 閾值）",
            meta=json.dumps({"remaining_hours": remaining, "threshold": QUOTA_WARN_THRESHOLD}),
        )
        db.add(row)
        db.commit()

        from notifications import notify, CTO_TARGET
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(notify(
                    CTO_TARGET,
                    f"[ModelHub] Lightning 配額預警：本月剩餘 {remaining:.1f} 小時，"
                    f"低於 {QUOTA_WARN_THRESHOLD}h 閾值。",
                ))
            else:
                loop.run_until_complete(notify(
                    CTO_TARGET,
                    f"[ModelHub] Lightning 配額預警：本月剩餘 {remaining:.1f} 小時。",
                ))
        except Exception:
            pass
        logger.warning("Lightning quota warning sent: remaining=%.1fh", remaining)
    except Exception as e:
        logger.warning("_check_lightning_quota_warning failed: %s", e)


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
        # Sprint 22 Task 22-4: 每次 poll 時檢查 Lightning 配額預警
        _check_lightning_quota_warning(db)

        rows = (
            db.query(Submission)
            .filter(
                Submission.training_resource == "lightning",
                Submission.status.in_(["training", "queued"]),
            )
            .all()
        )
        summary = {"checked": len(rows), "changed": 0, "complete": 0, "error": 0}
        for sub in rows:
            prev_status = sub.kaggle_status
            _process_submission(db, sub)
            # 重新讀取確認狀態是否有變化
            db.refresh(sub)
            if sub.kaggle_status != prev_status:
                summary["changed"] += 1
            if sub.status == "trained":
                summary["complete"] += 1
            elif sub.status == "failed":
                summary["error"] += 1
        logger.debug("Lightning poller tick: %s", summary)
        return summary
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
