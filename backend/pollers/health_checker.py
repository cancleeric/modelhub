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
# P1-4: stuck watchdog 自動 fail 開關（預設 true；dev 可設為 false 避免誤觸）
STUCK_AUTO_FAIL_ENABLED = os.environ.get("STUCK_AUTO_FAIL_ENABLED", "true").lower() == "true"

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
    """5. training 任務卡頓 > 30 小時 → 自動 mark_failed + notify CTO（每任務獨立節流）"""
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

            # P1-4: STUCK_AUTO_FAIL_ENABLED → 自動 mark_failed，不只發通知
            if STUCK_AUTO_FAIL_ENABLED:
                error_msg = f"stuck auto-failed after {elapsed_h:.1f}h (threshold={TRAINING_STUCK_HOURS}h)"
                sub.status = "failed"
                sub.error_message = error_msg if hasattr(sub, "error_message") else None

                # 呼叫 QueueManager.mark_failed_by_req
                try:
                    from queue_manager import QueueManager
                    QueueManager.mark_failed_by_req(db, sub.req_no, reason="stuck auto-failed")
                except Exception as _qe:
                    logger.warning("_check_stuck_trainings: queue mark_failed failed for %s: %s", sub.req_no, _qe)

                # 寫 SubmissionHistory
                _record_alert(
                    db, alert_type,
                    f"訓練任務 {sub.req_no} 已卡頓 {elapsed_h:.1f}h，自動標記失敗",
                    {"req_no": sub.req_no, "elapsed_hours": elapsed_h, "action": "auto_failed"},
                )
                row = SubmissionHistory(
                    req_no=sub.req_no,
                    action="stuck_auto_failed",
                    actor="health-checker",
                    note=error_msg,
                    meta=json.dumps({"elapsed_hours": elapsed_h, "threshold_hours": TRAINING_STUCK_HOURS},
                                    ensure_ascii=False),
                )
                db.add(row)
                db.commit()

                note = f"訓練任務 {sub.req_no} 已卡頓 {elapsed_h:.1f}h，已自動標記失敗（閾值 {TRAINING_STUCK_HOURS}h）"
                await _notify_cto(f"[ModelHub Health] {note}（已自動處理）")
                logger.warning("health_check: stuck_auto_failed req=%s elapsed=%.1fh", sub.req_no, elapsed_h)
            else:
                # STUCK_AUTO_FAIL_ENABLED=false：僅通知，不動作（舊行為）
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


def record_resource_all_exhausted(
    db,
    req_no: str,
    attempted: list,
    note: Optional[str] = None,
) -> None:
    """
    M18-4: 三個 resource（kaggle/lightning/ssh）都不可用時，
    寫入 events 表（event_type=resource_all_exhausted）。
    不走 CMC；UI 後續 phase 從 events 表顯示 toast / log。

    1 小時節流：同一 req_no 在過去 1 小時內已有相同 event 則跳過。
    """
    from datetime import timedelta
    from models import SystemEvent

    # 節流：查 events 表
    try:
        cutoff = datetime.utcnow() - timedelta(hours=1)
        existing = (
            db.query(SystemEvent)
            .filter(
                SystemEvent.event_type == "resource_all_exhausted",
                SystemEvent.req_no == req_no,
                SystemEvent.created_at >= cutoff,
            )
            .first()
        )
        if existing:
            logger.debug(
                "record_resource_all_exhausted: throttled req=%s (already recorded within 1h)",
                req_no,
            )
            return
    except Exception as _te:
        logger.warning("record_resource_all_exhausted: throttle check failed: %s", _te)

    # 寫 events 表
    try:
        message = (
            note
            or f"所有 training resource 均不可用（已嘗試：{', '.join(attempted)}），工單 {req_no} 標記失敗"
        )
        event = SystemEvent(
            event_type="resource_all_exhausted",
            req_no=req_no,
            severity="error",
            message=message,
            meta=json.dumps({"attempted": attempted, "req_no": req_no}, ensure_ascii=False),
            created_at=datetime.utcnow(),
        )
        db.add(event)
        db.commit()
        logger.warning(
            "resource_all_exhausted: req=%s attempted=%s event written to events table",
            req_no, attempted,
        )
    except Exception as e:
        logger.warning("record_resource_all_exhausted: failed to write event: %s", e)


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
    """每日 08:00 Asia/Taipei 發送健康日報（P3-29: 加結構化 Markdown 表格）"""
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

        # P3-29: 工單狀態分佈
        status_dist: dict = {}
        try:
            from sqlalchemy import func
            rows_dist = (
                db.query(Submission.status, func.count(Submission.id).label("cnt"))
                .group_by(Submission.status)
                .all()
            )
            status_dist = {r.status: r.cnt for r in rows_dist}
        except Exception:
            pass

        # P3-29: Lightning 最後 poll 時間
        lightning_last_poll_str = "未知"
        try:
            from pollers.lightning_poller import get_last_poll_at
            lp = get_last_poll_at()
            if lp:
                lightning_last_poll_str = lp.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass

        # P3-29: Kaggle 配額使用率
        kaggle_used_pct = "未知"
        try:
            if kaggle_remaining is not None:
                total_weekly = 30.0
                used = max(0.0, total_weekly - kaggle_remaining)
                kaggle_used_pct = f"{used / total_weekly * 100:.1f}%（已用 {used:.1f}h / {total_weekly:.0f}h）"
        except Exception:
            pass

        # 建構 Markdown 報告
        kaggle_str = f"{kaggle_remaining:.1f}h" if kaggle_remaining is not None else "未知"
        lightning_str = f"{lightning_remaining:.1f}h" if lightning_remaining is not None else "未知"

        # 工單狀態表格
        status_rows = "\n".join(
            f"| {s} | {c} |"
            for s, c in sorted(status_dist.items(), key=lambda x: -x[1])
        ) if status_dist else "| （無資料） | — |"

        lines = [
            "## [ModelHub] 每日健康報告",
            f"**時間：** {now.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "### 資源配額",
            "| 資源 | 狀態 |",
            "|------|------|",
            f"| Kaggle 本週剩餘 | {kaggle_str} |",
            f"| Kaggle 使用率 | {kaggle_used_pct} |",
            f"| Lightning 本月剩餘 | {lightning_str} |",
            f"| Lightning 最後 poll | {lightning_last_poll_str} |",
            "",
            "### 訓練隊列",
            "| 指標 | 數值 |",
            "|------|------|",
            f"| 訓練中任務 | {active_count} 件 |",
            f"| 隊列 waiting | {waiting_count} |",
            f"| 隊列 running | {running_count} |",
            "",
            "### 工單狀態分佈",
            "| 狀態 | 件數 |",
            "|------|------|",
            status_rows,
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
