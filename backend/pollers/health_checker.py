"""
pollers/health_checker.py — 系統健康定時掃描（Sprint 21）

每 10 分鐘執行，掃描 6 種警告條件：
1. Kaggle 配額 < 5h
2. Lightning 配額 < 3h
3. Kaggle poller 最後執行 > 5 分鐘
4. Lightning poller 最後執行 > 15 分鐘
5. Training 任務卡頓 > 30 小時
6. waiting > 5 筆且無 running

每種警告類型 1 小時節流（記錄在 SubmissionHistory req_no="__health__"）。
每日 08:00 Asia/Taipei 發送健康日報。
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from models import SessionLocal, Submission, SubmissionHistory

logger = logging.getLogger("modelhub.poller.health_checker")

CHECK_INTERVAL_SECONDS = int(os.environ.get("MODELHUB_HEALTH_CHECK_INTERVAL", "600"))  # 10 min
KAGGLE_QUOTA_WARN_HOURS = float(os.environ.get("KAGGLE_QUOTA_WARN_HOURS", "5"))
LIGHTNING_QUOTA_WARN_HOURS = float(os.environ.get("LIGHTNING_QUOTA_WARN_HOURS", "3"))
KAGGLE_POLLER_STALE_MINUTES = int(os.environ.get("KAGGLE_POLLER_STALE_MINUTES", "5"))
LIGHTNING_POLLER_STALE_MINUTES = int(os.environ.get("LIGHTNING_POLLER_STALE_MINUTES", "15"))
TRAINING_STUCK_HOURS = int(os.environ.get("TRAINING_STUCK_HOURS", "30"))
QUEUE_WARN_WAITING_COUNT = int(os.environ.get("QUEUE_WARN_WAITING_COUNT", "5"))

_scheduler = None
_last_check_at: Optional[datetime] = None


def get_last_check_at() -> Optional[datetime]:
    return _last_check_at


def _throttle_key(alert_type: str) -> bool:
    """
    檢查 1 小時節流：若過去 1 小時內已有同 alert_type 的 history，return True（已節流）。
    """
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(hours=1)
        existing = (
            db.query(SubmissionHistory)
            .filter(
                SubmissionHistory.req_no == "__health__",
                SubmissionHistory.action == alert_type,
                SubmissionHistory.created_at >= cutoff,
            )
            .first()
        )
        return existing is not None
    finally:
        db.close()


def _record_alert(db, alert_type: str, note: str, meta: Optional[dict] = None) -> None:
    """寫入 SubmissionHistory 做節流記錄"""
    row = SubmissionHistory(
        req_no="__health__",
        action=alert_type,
        actor="health-checker",
        note=note,
        meta=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(row)
    db.commit()


async def _notify_cto(message: str) -> None:
    from notifications import notify, CTO_TARGET
    await notify(CTO_TARGET, message)


async def _check_kaggle_quota(db) -> None:
    """1. Kaggle 配額 < 5h → notify CTO"""
    alert_type = "health_kaggle_quota_low"
    if _throttle_key(alert_type):
        return
    try:
        from resources.prober import KaggleQuotaTracker
        tracker = KaggleQuotaTracker()
        remaining = tracker.get_remaining_hours(db)
        if remaining >= KAGGLE_QUOTA_WARN_HOURS:
            return
        note = f"Kaggle 本週剩餘 {remaining:.1f}h（閾值 {KAGGLE_QUOTA_WARN_HOURS}h）"
        _record_alert(db, alert_type, note, {"remaining_hours": remaining})
        await _notify_cto(f"[ModelHub Health] {note}")
        logger.warning("health_check: kaggle_quota_low remaining=%.1fh", remaining)
    except Exception as e:
        logger.warning("health_check: kaggle quota check failed: %s", e)


async def _check_lightning_quota(db) -> None:
    """2. Lightning 配額 < 3h → notify CTO"""
    alert_type = "health_lightning_quota_low"
    if _throttle_key(alert_type):
        return
    try:
        today = datetime.utcnow()
        month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        rows = (
            db.query(Submission)
            .filter(
                Submission.training_resource == "lightning",
                Submission.training_started_at >= month_start,
                Submission.gpu_seconds.isnot(None),
            )
            .all()
        )
        used_sec = sum(r.gpu_seconds for r in rows if r.gpu_seconds)
        used_h = used_sec / 3600.0
        remaining = max(0.0, 22.0 - used_h)
        if remaining >= LIGHTNING_QUOTA_WARN_HOURS:
            return
        note = f"Lightning 本月剩餘 {remaining:.1f}h（閾值 {LIGHTNING_QUOTA_WARN_HOURS}h）"
        _record_alert(db, alert_type, note, {"remaining_hours": remaining})
        await _notify_cto(f"[ModelHub Health] {note}")
        logger.warning("health_check: lightning_quota_low remaining=%.1fh", remaining)
    except Exception as e:
        logger.warning("health_check: lightning quota check failed: %s", e)


async def _check_kaggle_poller(db) -> None:
    """3. Kaggle poller 最後執行 > 5 分鐘 → notify CTO"""
    alert_type = "health_kaggle_poller_stale"
    if _throttle_key(alert_type):
        return
    try:
        from pollers.kaggle_poller import get_last_poll_at
        last_at = get_last_poll_at()
        if last_at is None:
            note = "Kaggle poller 從未執行"
        else:
            elapsed_min = (datetime.utcnow() - last_at).total_seconds() / 60
            if elapsed_min < KAGGLE_POLLER_STALE_MINUTES:
                return
            note = f"Kaggle poller 最後執行 {elapsed_min:.0f} 分鐘前（閾值 {KAGGLE_POLLER_STALE_MINUTES} 分鐘）"
        _record_alert(db, alert_type, note)
        await _notify_cto(f"[ModelHub Health] {note}")
        logger.warning("health_check: kaggle_poller_stale")
    except Exception as e:
        logger.warning("health_check: kaggle poller check failed: %s", e)


async def _check_lightning_poller(db) -> None:
    """4. Lightning poller 最後執行 > 15 分鐘 → notify CTO"""
    alert_type = "health_lightning_poller_stale"
    if _throttle_key(alert_type):
        return
    try:
        from pollers.lightning_poller import get_last_poll_at
        last_at = get_last_poll_at()
        if last_at is None:
            note = "Lightning poller 從未執行"
        else:
            elapsed_min = (datetime.utcnow() - last_at).total_seconds() / 60
            if elapsed_min < LIGHTNING_POLLER_STALE_MINUTES:
                return
            note = f"Lightning poller 最後執行 {elapsed_min:.0f} 分鐘前（閾值 {LIGHTNING_POLLER_STALE_MINUTES} 分鐘）"
        _record_alert(db, alert_type, note)
        await _notify_cto(f"[ModelHub Health] {note}")
        logger.warning("health_check: lightning_poller_stale")
    except Exception as e:
        logger.warning("health_check: lightning poller check failed: %s", e)


async def _check_stuck_trainings(db) -> None:
    """5. training 任務卡頓 > 30 小時 → notify CTO（每任務獨立節流）"""
    try:
        cutoff = datetime.utcnow() - timedelta(hours=TRAINING_STUCK_HOURS)
        stuck = (
            db.query(Submission)
            .filter(
                Submission.status == "training",
                Submission.training_started_at.isnot(None),
                Submission.training_started_at < cutoff,
            )
            .all()
        )
        for sub in stuck:
            alert_type = f"health_stuck_{sub.req_no}"
            if _throttle_key(alert_type):
                continue
            elapsed_h = (datetime.utcnow() - sub.training_started_at).total_seconds() / 3600
            note = f"訓練任務 {sub.req_no} 已卡頓 {elapsed_h:.1f}h（閾值 {TRAINING_STUCK_HOURS}h）"
            _record_alert(db, alert_type, note, {"req_no": sub.req_no, "elapsed_hours": elapsed_h})
            await _notify_cto(f"[ModelHub Health] {note}")
            logger.warning("health_check: stuck training req=%s elapsed=%.1fh", sub.req_no, elapsed_h)
    except Exception as e:
        logger.warning("health_check: stuck training check failed: %s", e)


async def _check_queue_starvation(db) -> None:
    """6. waiting > 5 筆且無 running → notify CTO"""
    alert_type = "health_queue_starvation"
    if _throttle_key(alert_type):
        return
    try:
        from queue_manager import QueueManager
        waiting_count = len(QueueManager.get_all_waiting(db))
        running_count = QueueManager.count_running(db)
        if waiting_count <= QUEUE_WARN_WAITING_COUNT or running_count > 0:
            return
        note = f"隊列積壓 {waiting_count} 筆，且無任何任務在執行中"
        _record_alert(db, alert_type, note, {"waiting_count": waiting_count, "running_count": running_count})
        await _notify_cto(f"[ModelHub Health] {note}")
        logger.warning("health_check: queue_starvation waiting=%d running=%d", waiting_count, running_count)
    except Exception as e:
        logger.warning("health_check: queue starvation check failed: %s", e)


async def run_health_check() -> dict:
    """執行所有健康檢查（async，供 APScheduler AsyncIOScheduler 呼叫）"""
    global _last_check_at
    _last_check_at = datetime.utcnow()

    db = SessionLocal()
    results = {}
    try:
        await _check_kaggle_quota(db)
        results["kaggle_quota"] = "checked"
        await _check_lightning_quota(db)
        results["lightning_quota"] = "checked"
        await _check_kaggle_poller(db)
        results["kaggle_poller"] = "checked"
        await _check_lightning_poller(db)
        results["lightning_poller"] = "checked"
        await _check_stuck_trainings(db)
        results["stuck_trainings"] = "checked"
        await _check_queue_starvation(db)
        results["queue_starvation"] = "checked"
        logger.debug("health_check completed: %s", results)
        return results
    except Exception as e:
        logger.exception("health_check exception: %s", e)
        return {"error": str(e)}
    finally:
        db.close()


async def send_daily_report() -> None:
    """每日 08:00 Asia/Taipei 發送健康日報"""
    from notifications import notify, CTO_TARGET
    from queue_manager import QueueManager

    db = SessionLocal()
    try:
        now = datetime.utcnow()

        # Kaggle 配額
        kaggle_remaining = None
        try:
            from resources.prober import KaggleQuotaTracker
            tracker = KaggleQuotaTracker()
            kaggle_remaining = tracker.get_remaining_hours(db)
        except Exception:
            pass

        # Lightning 配額
        lightning_remaining = None
        try:
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            rows = (
                db.query(Submission)
                .filter(
                    Submission.training_resource == "lightning",
                    Submission.training_started_at >= month_start,
                    Submission.gpu_seconds.isnot(None),
                )
                .all()
            )
            used_sec = sum(r.gpu_seconds for r in rows if r.gpu_seconds)
            lightning_remaining = round(max(0.0, 22.0 - used_sec / 3600.0), 2)
        except Exception:
            pass

        # 隊列狀態
        waiting_count = 0
        running_count = 0
        try:
            waiting_count = len(QueueManager.get_all_waiting(db))
            running_count = QueueManager.count_running(db)
        except Exception:
            pass

        # 活躍訓練
        active_count = 0
        try:
            active_count = (
                db.query(Submission)
                .filter(Submission.status.in_(["training", "queued"]))
                .count()
            )
        except Exception:
            pass

        lines = [
            "[ModelHub] 每日健康報告",
            f"時間：{now.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"Kaggle 本週剩餘：{f'{kaggle_remaining:.1f}h' if kaggle_remaining is not None else '未知'}",
            f"Lightning 本月剩餘：{f'{lightning_remaining:.1f}h' if lightning_remaining is not None else '未知'}",
            f"訓練中任務：{active_count} 件",
            f"隊列 waiting：{waiting_count} / running：{running_count}",
        ]

        await notify(CTO_TARGET, "\n".join(lines))
        logger.info("daily_report sent to %s", CTO_TARGET)
    except Exception as e:
        logger.warning("send_daily_report failed: %s", e)
    finally:
        db.close()


def start_scheduler():
    global _scheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        logger.warning("apscheduler not installed, health checker disabled")
        return None

    if _scheduler:
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone="Asia/Taipei")
    _scheduler.add_job(
        run_health_check, "interval",
        seconds=CHECK_INTERVAL_SECONDS,
        id="health-checker",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.add_job(
        send_daily_report, "cron",
        hour=8, minute=0,
        id="daily-health-report",
    )
    _scheduler.start()
    logger.info("Health checker started (interval=%ds)", CHECK_INTERVAL_SECONDS)
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Health checker stopped")
