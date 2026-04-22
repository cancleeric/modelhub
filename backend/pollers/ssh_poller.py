"""
pollers/ssh_poller.py — SSH 訓練狀態 poller（Sprint 23 Task 23-3）

每 120 秒掃一輪 training_resource 以 "ssh@" 開頭且 status=training 的 submission：
- 查詢遠端 job 狀態
- complete → 下載 output → 解析 metrics → 建 ModelVersion → status=trained
- error → 標記失敗
- overtime > 30h → notify CTO

架構與 lightning_poller 一致：start_scheduler / stop_scheduler / poll_once。
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from models import SessionLocal, Submission, ModelVersion, SubmissionHistory
from parsers import parse_training_log
from notifications import notify_event
from utils import next_version_for as _next_version_for, read_log_files as _read_log_files

logger = logging.getLogger("modelhub.poller.ssh")

POLL_INTERVAL_SECONDS = int(os.environ.get("MODELHUB_SSH_POLL_INTERVAL", "120"))
OVERTIME_HOURS = int(os.environ.get("MODELHUB_SSH_OVERTIME_HOURS", "30"))
SSH_DOWNLOAD_DIR = os.environ.get("MODELHUB_SSH_DL_DIR", "/tmp/modelhub-ssh")

_scheduler = None
_last_poll_at: Optional[datetime] = None


def get_last_poll_at() -> Optional[datetime]:
    return _last_poll_at


def _append_history(db: Session, req_no: str, action: str, note: Optional[str] = None,
                    meta: Optional[dict] = None) -> None:
    row = SubmissionHistory(
        req_no=req_no,
        action=action,
        actor="ssh-poller",
        note=note,
        meta=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(row)


def _get_host_from_resource(training_resource: str) -> Optional[str]:
    """從 training_resource（ssh@user@host）解析主機部分"""
    if not training_resource or not training_resource.startswith("ssh@"):
        return None
    return training_resource[len("ssh@"):]


# P3-28: _read_log_files 已移至 utils.read_log_files，透過 import 取得


def _process_submission(db: Session, sub: Submission) -> None:
    """處理單一 SSH submission 的 poll 週期"""
    host = _get_host_from_resource(sub.training_resource or "")
    if not host:
        logger.warning("_process_submission: req=%s has no host in training_resource=%s",
                       sub.req_no, sub.training_resource)
        return

    from resources.ssh_launcher import SSHLauncher
    launcher = SSHLauncher()

    status = launcher.get_job_status(host, sub.req_no)
    logger.info("ssh_poller: req=%s host=%s status=%s", sub.req_no, host, status)

    # 狀態變化寫 history
    if sub.kaggle_status != status:
        sub.kaggle_status = status
        sub.kaggle_status_updated_at = datetime.utcnow()
        _append_history(db, req_no=sub.req_no, action="ssh_status_change",
                        meta={"new_status": status, "host": host})
        db.commit()

    if status == "complete":
        _on_ssh_complete(db, sub, host, launcher)
    elif status == "error":
        _on_ssh_error(db, sub)
    else:
        # 檢查 overtime
        if sub.training_started_at:
            elapsed = datetime.utcnow() - sub.training_started_at
            if elapsed > timedelta(hours=OVERTIME_HOURS):
                _append_history(db, req_no=sub.req_no, action="training_overtime",
                                meta={"elapsed_hours": elapsed.total_seconds() / 3600, "host": host})
                db.commit()
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.run_coroutine_threadsafe(notify_event("training_overtime", sub), loop)
                except RuntimeError:
                    asyncio.run(notify_event("training_overtime", sub))
                except Exception:
                    pass


def _on_ssh_complete(db: Session, sub: Submission, host: str, launcher) -> None:
    """SSH 任務完成處理"""
    # 幂等 guard
    if sub.status in ("trained", "accepted", "rejected"):
        return

    dest_dir = str(Path(SSH_DOWNLOAD_DIR) / sub.req_no)
    output_ok = launcher.download_output(host, sub.req_no, dest_dir)

    metrics = {}
    parsed = {}
    if output_ok:
        log_text = _read_log_files(dest_dir)
        if log_text:
            parsed = parse_training_log(sub.arch or "yolov8m", log_text)
            metrics = parsed.get("metrics", {}) or {}

    now = datetime.utcnow()
    sub.status = "trained"
    sub.training_completed_at = now
    sub.kaggle_status = "complete"
    sub.kaggle_status_updated_at = now

    if sub.training_started_at:
        gpu_seconds = int((now - sub.training_started_at).total_seconds())
        sub.gpu_seconds = gpu_seconds
        sub.estimated_cost_usd = 0.0  # SSH 內網 GPU 不計費

    per_class = parsed.get("per_class") if parsed else None
    if per_class and isinstance(per_class, dict):
        try:
            sub.per_class_metrics = json.dumps(per_class, ensure_ascii=False)
        except Exception:
            pass

    next_ver = _next_version_for(db, sub.req_no)
    mv = ModelVersion(
        req_no=sub.req_no,
        product=sub.product,
        model_name=sub.req_name or f"{sub.product} model",
        version=next_ver,
        train_date=now.strftime("%Y-%m-%d"),
        map50=metrics.get("map50"),
        map50_95=metrics.get("map50_95"),
        map50_actual=metrics.get("map50"),
        map50_95_actual=metrics.get("map50_95"),
        epochs=metrics.get("epochs"),
        batch_size=metrics.get("batch_size"),
        arch=sub.arch,
        status="pending_acceptance",
        notes=f"auto-filled by ssh-poller at {now.isoformat()}",
    )
    db.add(mv)

    _append_history(db, req_no=sub.req_no, action="training_complete",
                    meta={"metrics": metrics, "gpu_seconds": sub.gpu_seconds,
                          "version": next_ver, "platform": f"ssh@{host}"})
    db.commit()
    db.refresh(sub)

    # P2-20: 完成後通知 QueueManager
    try:
        from queue_manager import QueueManager
        db2 = SessionLocal()
        try:
            QueueManager.mark_done_by_req(db2, sub.req_no)
            db2.commit()
        finally:
            db2.close()
    except Exception as _qe:
        logger.warning("_on_ssh_complete: queue mark_done failed: %s", _qe)

    import asyncio
    try:
        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(notify_event("training_complete", sub), loop)
    except RuntimeError:
        asyncio.run(notify_event("training_complete", sub))
    except Exception:
        pass


def _on_ssh_error(db: Session, sub: Submission) -> None:
    """SSH 任務失敗處理"""
    now = datetime.utcnow()
    sub.status = "failed"
    sub.training_completed_at = now
    if sub.training_started_at:
        sub.gpu_seconds = int((now - sub.training_started_at).total_seconds())

    _append_history(db, req_no=sub.req_no, action="training_failed",
                    meta={"platform": sub.training_resource})
    db.commit()

    import asyncio
    try:
        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(notify_event("training_failed", sub), loop)
    except RuntimeError:
        asyncio.run(notify_event("training_failed", sub))
    except Exception:
        pass

    # 更新 queue 狀態
    try:
        from queue_manager import QueueManager
        db2 = SessionLocal()
        try:
            QueueManager.mark_failed_by_req(db2, sub.req_no, "ssh training error")
            db2.commit()
        finally:
            db2.close()
    except Exception as _qe:
        logger.warning("_on_ssh_error: queue mark_failed failed: %s", _qe)


def poll_once() -> dict:
    """掃一輪 SSH 訓練 submission（同步，供 APScheduler 呼叫）"""
    global _last_poll_at
    _last_poll_at = datetime.utcnow()

    db: Session = SessionLocal()
    try:
        rows = (
            db.query(Submission)
            .filter(
                Submission.training_resource.like("ssh@%"),
                Submission.status.in_(["training", "queued"]),
            )
            .all()
        )
        summary = {"checked": len(rows), "changed": 0, "complete": 0, "error": 0}

        for sub in rows:
            prev_status = sub.status
            _process_submission(db, sub)
            db.refresh(sub)
            if sub.status != prev_status:
                summary["changed"] += 1
            if sub.status == "trained":
                summary["complete"] += 1
            elif sub.status == "failed":
                summary["error"] += 1

        logger.debug("SSH poller tick: %s", summary)
        return summary

    except Exception as exc:
        logger.exception("SSH poller poll_once exception: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()


def start_scheduler():
    global _scheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        logger.warning("apscheduler not installed, SSH poller disabled")
        return None

    if _scheduler:
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone="Asia/Taipei")
    _scheduler.add_job(
        poll_once, "interval",
        seconds=POLL_INTERVAL_SECONDS,
        id="ssh-poller",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info("SSH poller started (interval=%ds)", POLL_INTERVAL_SECONDS)
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("SSH poller stopped")
