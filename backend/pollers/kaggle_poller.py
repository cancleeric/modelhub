"""
Kaggle Kernel poller — Sprint 3 / 4

每 60 秒 poll 所有 status=training 的 submission：
- 狀態變化寫 submission_history
- complete 時下載 output → 解析 metrics → 建 ModelVersion
- error 時標記 training_failed
- >24 小時未完成 → 通知 CTO overtime

kaggle SDK 需要 KAGGLE_USERNAME + KAGGLE_KEY env var。
如果 env var 未設，poller 還是會啟動但會 log warning，並跳過 API 呼叫。
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from models import SessionLocal, Submission, ModelVersion
from parsers import parse_training_log
from notifications import notify_event

logger = logging.getLogger("modelhub.poller.kaggle")

POLL_INTERVAL_SECONDS = int(os.environ.get("MODELHUB_KAGGLE_POLL_INTERVAL", "60"))
OVERTIME_HOURS = int(os.environ.get("MODELHUB_TRAINING_OVERTIME_HOURS", "24"))
KAGGLE_DOWNLOAD_DIR = os.environ.get("MODELHUB_KAGGLE_DL_DIR", "/tmp/modelhub-kaggle")

# Kaggle P100 成本估算（$0.5/hr）
GPU_USD_PER_HOUR = float(os.environ.get("MODELHUB_GPU_USD_PER_HOUR", "0.5"))

# Sprint 15 P0-3: Kaggle 免費配額時不計費
# KAGGLE_IS_PAID_TIER=false（預設）→ estimated_cost_usd=0，仍記錄 gpu_seconds
KAGGLE_IS_PAID_TIER = os.environ.get("KAGGLE_IS_PAID_TIER", "false").lower() == "true"


def _kaggle_env_ready() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def _get_kaggle_api():
    """Lazy import kaggle SDK（authenticate 需要 env 已設）"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        logger.warning("Kaggle SDK unavailable: %s", e)
        return None


async def _fetch_kernel_status(api, slug: str) -> Optional[dict]:
    """
    呼叫 kaggle kernels status <slug>
    kaggle SDK 沒有純 Python status API，用 subprocess 最穩。
    """
    if not shutil.which("kaggle"):
        logger.warning("kaggle CLI not in PATH, skipping status fetch")
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "kaggle", "kernels", "status", slug,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode != 0:
            logger.warning("kaggle status fail (slug=%s): %s",
                           slug, stderr.decode(errors="replace"))
            return None
        out = stdout.decode(errors="replace")
        # 預期輸出樣例：`<slug> has status "complete"` / "running" / "error" / "queued"
        m = re.search(r'status\s+"?(\w+)"?', out)
        if not m:
            return None
        return {"status": m.group(1).lower(), "raw": out.strip()}
    except Exception as e:
        logger.warning("kaggle status exception (slug=%s): %s", slug, e)
        return None


async def _download_kernel_output(slug: str, dest_dir: str) -> Optional[str]:
    """下載 kernel 的 output log，回傳本地資料夾路徑"""
    if not shutil.which("kaggle"):
        return None
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    try:
        proc = await asyncio.create_subprocess_exec(
            "kaggle", "kernels", "output", slug, "-p", dest_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        if proc.returncode != 0:
            logger.warning("kaggle output fail (slug=%s): %s",
                           slug, stderr.decode(errors="replace"))
            return None
        return dest_dir
    except Exception as e:
        logger.warning("kaggle output exception (slug=%s): %s", slug, e)
        return None


def _read_log_files(dir_path: str) -> str:
    """把 dir 裡所有 .log / .txt / .json 都 concat（不超過 5MB）"""
    acc = []
    MAX_SIZE = 5 * 1024 * 1024
    total = 0
    root = Path(dir_path)
    if not root.exists():
        return ""
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".log", ".txt", ".json", ".out", ".stdout"):
            continue
        try:
            size = f.stat().st_size
            if total + size > MAX_SIZE:
                break
            acc.append(f.read_text(errors="replace"))
            total += size
        except Exception:
            continue
    return "\n".join(acc)


def _append_history(db: Session, req_no: str, action: str, meta: Optional[dict] = None,
                    note: Optional[str] = None) -> None:
    from models import SubmissionHistory
    row = SubmissionHistory(
        req_no=req_no,
        action=action,
        actor="kaggle-poller",
        note=note,
        meta=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(row)


async def _on_kernel_complete(db: Session, sub: Submission) -> None:
    """complete → 下載 output → 解析 → 建 ModelVersion → status=trained"""
    if not sub.kaggle_kernel_slug:
        return
    dest = os.path.join(KAGGLE_DOWNLOAD_DIR, sub.req_no)
    downloaded = await _download_kernel_output(sub.kaggle_kernel_slug, dest)
    metrics: dict = {}
    parsed: dict = {}
    log_text = ""
    if downloaded:
        log_text = _read_log_files(downloaded)
        if log_text:
            parsed = parse_training_log(sub.arch, log_text)
            metrics = parsed.get("metrics", {}) or {}

    now = datetime.utcnow()
    sub.kaggle_status = "complete"
    sub.kaggle_status_updated_at = now
    sub.training_completed_at = now
    sub.status = "trained"

    # Sprint 4: GPU 時數估算（start → now）
    # Sprint 15 P0-3: 免費配額時 cost=0，仍記錄 gpu_seconds
    if sub.training_started_at:
        gpu_seconds = int((now - sub.training_started_at).total_seconds())
        sub.gpu_seconds = gpu_seconds
        if KAGGLE_IS_PAID_TIER:
            sub.estimated_cost_usd = round(gpu_seconds / 3600 * GPU_USD_PER_HOUR, 4)
        else:
            sub.estimated_cost_usd = 0.0

    # 自動建 ModelVersion（pending_acceptance）
    next_version = _next_version_for(db, sub.req_no)

    # P3-4: 若有 per_class 指標，同步寫入 submission 並帶入 mv notes
    per_class = parsed.get("per_class") or parsed.get("class_metrics") if downloaded and log_text else None
    if per_class and isinstance(per_class, dict):
        import json as _json_pc
        try:
            sub.per_class_metrics = _json_pc.dumps(per_class, ensure_ascii=False)
        except Exception:
            pass

    mv = ModelVersion(
        req_no=sub.req_no,
        product=sub.product,
        model_name=sub.req_name or f"{sub.product} model",
        version=next_version,
        train_date=now.strftime("%Y-%m-%d"),
        map50=metrics.get("map50"),
        map50_95=metrics.get("map50_95"),
        map50_actual=metrics.get("map50"),
        map50_95_actual=metrics.get("map50_95"),
        epochs=metrics.get("epochs"),
        batch_size=metrics.get("batch_size"),
        arch=sub.arch,
        kaggle_kernel_url=f"https://www.kaggle.com/code/{sub.kaggle_kernel_slug}",
        status="pending_acceptance",
        notes=f"auto-filled by kaggle-poller at {now.isoformat()}",
    )
    db.add(mv)

    _append_history(
        db,
        req_no=sub.req_no,
        action="training_complete",
        meta={
            "metrics": metrics,
            "gpu_seconds": sub.gpu_seconds,
            "estimated_cost_usd": sub.estimated_cost_usd,
            "version": next_version,
        },
    )
    db.commit()
    db.refresh(sub)
    await notify_event("training_complete", sub)


def _next_version_for(db: Session, req_no: str) -> str:
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


async def _on_kernel_error(db: Session, sub: Submission, raw: str) -> None:
    now = datetime.utcnow()
    sub.kaggle_status = "error"
    sub.kaggle_status_updated_at = now

    retry_count = sub.retry_count or 0
    max_retries = sub.max_retries if sub.max_retries is not None else 2

    if retry_count < max_retries:
        # Sprint 6.4: 自動重試。狀態保持 training，不通知失敗。
        sub.retry_count = retry_count + 1
        _append_history(
            db,
            req_no=sub.req_no,
            action="training_retry",
            note=(raw[:300] if raw else None),
            meta={"retry_count": sub.retry_count, "max_retries": max_retries},
        )
        db.commit()
        pushed = await _push_kernel(sub.kaggle_kernel_slug)
        logger.info("auto-retry req=%s slug=%s push=%s",
                    sub.req_no, sub.kaggle_kernel_slug, pushed)
        # 下一輪 poll 會看到 queued/running，不在此通知
        sub.kaggle_status = "queued"
        db.commit()
        return

    # 已達上限 → 真 failed
    sub.training_completed_at = now
    sub.status = "failed"
    if sub.training_started_at:
        sub.gpu_seconds = int((now - sub.training_started_at).total_seconds())

    _append_history(
        db,
        req_no=sub.req_no,
        action="training_failed",
        note=raw[:500] if raw else None,
        meta={"retry_count": retry_count, "max_retries": max_retries},
    )
    db.commit()
    await notify_event("training_failed", sub)


async def _push_kernel(slug: str) -> bool:
    """重新 push kernel（觸發重跑）— 需要 kaggle CLI + 本地 metadata。"""
    import shutil
    if not slug or not shutil.which("kaggle"):
        return False
    # 若先前沒 pull 過，先 pull 一次把 metadata 抓下來
    dest = os.path.join(KAGGLE_DOWNLOAD_DIR, "retry", slug.replace("/", "_"))
    Path(dest).mkdir(parents=True, exist_ok=True)
    try:
        p = await asyncio.create_subprocess_exec(
            "kaggle", "kernels", "pull", slug, "-p", dest, "-m",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(p.communicate(), timeout=60.0)
        p = await asyncio.create_subprocess_exec(
            "kaggle", "kernels", "push", "-p", dest,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(p.communicate(), timeout=60.0)
        if p.returncode != 0:
            logger.warning("kaggle push fail (slug=%s): %s",
                           slug, stderr.decode(errors="replace"))
            return False
        return True
    except Exception as e:
        logger.warning("kaggle push exception (slug=%s): %s", slug, e)
        return False


async def _check_budget(db: Session, sub: Submission) -> None:
    """Sprint 6.7: estimated_cost > max_budget 時一次性通知 CTO。"""
    if sub.budget_exceeded_notified:
        return
    budget = sub.max_budget_usd or 0
    if budget <= 0:
        return

    # 估算當下成本：start → now
    if not sub.training_started_at:
        return
    elapsed_s = (datetime.utcnow() - sub.training_started_at).total_seconds()
    # Sprint 15 P0-3: 免費配額時 cost=0，不會觸發預算超限
    if not KAGGLE_IS_PAID_TIER:
        return
    current_cost = elapsed_s / 3600 * GPU_USD_PER_HOUR
    if current_cost <= budget:
        return
    sub.budget_exceeded_notified = True
    sub.estimated_cost_usd = round(current_cost, 4)
    _append_history(
        db, req_no=sub.req_no, action="budget_exceeded",
        meta={"current_cost_usd": round(current_cost, 4), "max_budget_usd": budget},
    )
    db.commit()
    try:
        from notifications import notify, CTO_TARGET
        await notify(
            CTO_TARGET,
            f"[ModelHub] 需求單 {sub.req_no} 訓練成本已超過預算上限 "
            f"(${current_cost:.2f} / ${budget:.2f})，請確認是否繼續。"
        )
    except Exception:
        pass


async def _check_overtime(db: Session, sub: Submission) -> None:
    if not sub.training_started_at:
        return
    elapsed = datetime.utcnow() - sub.training_started_at
    if elapsed < timedelta(hours=OVERTIME_HOURS):
        return
    # 每 submission 只通知一次（用 history 判）
    from models import SubmissionHistory
    already = (
        db.query(SubmissionHistory)
        .filter(
            SubmissionHistory.req_no == sub.req_no,
            SubmissionHistory.action == "training_overtime",
        )
        .first()
    )
    if already:
        return
    _append_history(db, req_no=sub.req_no, action="training_overtime",
                    meta={"elapsed_hours": elapsed.total_seconds() / 3600})
    db.commit()
    await notify_event("training_overtime", sub)


async def _check_quota_warning(db: Session) -> None:
    """
    P1-2: Kaggle 配額預警。
    剩 < 5h 且本週還沒發過預警時，寫 history + notify_event。
    """
    QUOTA_WARN_THRESHOLD = float(os.environ.get("KAGGLE_QUOTA_WARN_HOURS", "5"))
    try:
        from resources.prober import KaggleQuotaTracker
        tracker = KaggleQuotaTracker()
        remaining = tracker.get_remaining_hours(db)
        if remaining >= QUOTA_WARN_THRESHOLD:
            return

        # 本週是否已發過預警（用 SubmissionHistory 記錄，req_no="__quota__"）
        from models import SubmissionHistory
        from datetime import timedelta
        today = datetime.utcnow()
        monday = today - timedelta(days=today.weekday())
        week_start = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        already = (
            db.query(SubmissionHistory)
            .filter(
                SubmissionHistory.req_no == "__quota__",
                SubmissionHistory.action == "kaggle_quota_warning",
                SubmissionHistory.created_at >= week_start,
            )
            .first()
        )
        if already:
            return

        # 寫 history 防重複
        row = SubmissionHistory(
            req_no="__quota__",
            action="kaggle_quota_warning",
            actor="kaggle-poller",
            note=f"本週剩餘配額 {remaining:.1f}h（低於 {QUOTA_WARN_THRESHOLD}h 閾值）",
            meta=json.dumps({"remaining_hours": remaining, "threshold": QUOTA_WARN_THRESHOLD},
                            ensure_ascii=False),
        )
        db.add(row)
        db.commit()

        # 通知（用 notify_event 的低階 notify 直接發給 CTO）
        from notifications import notify, CTO_TARGET
        await notify(
            CTO_TARGET,
            f"[ModelHub] Kaggle 配額預警：本週剩餘 {remaining:.1f} 小時，"
            f"低於 {QUOTA_WARN_THRESHOLD}h 閾值，請注意配額用量。",
        )
        logger.warning("Kaggle quota warning sent: remaining=%.1fh", remaining)
    except Exception as e:
        logger.warning("_check_quota_warning failed: %s", e)


async def poll_once() -> dict:
    """掃一輪 status=training 的 submission"""
    global _last_poll_at
    _last_poll_at = datetime.utcnow()

    if not _kaggle_env_ready():
        logger.debug("KAGGLE_USERNAME/KAGGLE_KEY not set, skip poll")
        return {"skipped": True}

    api = _get_kaggle_api()
    # 即使 api 取不到，subprocess 路徑仍可跑

    db: Session = SessionLocal()
    try:
        # P1-2: 每次 poll 時檢查配額預警
        await _check_quota_warning(db)

        rows = db.query(Submission).filter(Submission.status == "training").all()
        summary = {"checked": len(rows), "changed": 0, "complete": 0, "error": 0}

        for sub in rows:
            if not sub.kaggle_kernel_slug:
                continue
            await _check_overtime(db, sub)
            await _check_budget(db, sub)
            status_result = await _fetch_kernel_status(api, sub.kaggle_kernel_slug)
            if not status_result:
                continue
            new_status = status_result["status"]
            raw = status_result.get("raw", "")

            if new_status != sub.kaggle_status:
                summary["changed"] += 1
                sub.kaggle_status = new_status
                sub.kaggle_status_updated_at = datetime.utcnow()
                _append_history(
                    db, req_no=sub.req_no, action="kaggle_status_change",
                    meta={"new_status": new_status},
                )
                db.commit()

            if new_status == "complete":
                summary["complete"] += 1
                await _on_kernel_complete(db, sub)
            elif new_status == "error":
                summary["error"] += 1
                await _on_kernel_error(db, sub, raw)

        return summary
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
        logger.warning("apscheduler not installed, poller disabled")
        return None

    if _scheduler:
        return _scheduler
    _scheduler = AsyncIOScheduler(timezone="Asia/Taipei")
    _scheduler.add_job(
        poll_once, "interval",
        seconds=POLL_INTERVAL_SECONDS,
        id="kaggle-poller",
        max_instances=1,
        coalesce=True,
    )
    # 週報：每週一 09:00 Asia/Taipei
    from .weekly_report import send_weekly_report
    _scheduler.add_job(
        send_weekly_report, "cron",
        day_of_week="mon", hour=9, minute=0,
        id="weekly-report",
    )
    _scheduler.start()
    logger.info("Kaggle poller + weekly report scheduler started (interval=%ds)",
                POLL_INTERVAL_SECONDS)
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")
