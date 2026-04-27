"""
pollers/queue_dispatcher.py — 訓練隊列派發器（Sprint 20 Task 20-4）

每 30 秒執行一次，從持久化 TrainingQueue 取下一筆 waiting 條目並派發訓練。
MAX_CONCURRENT_TRAININGS=2 上限，超過時跳過本次。

M18: Cross-Resource Auto Retry
- dispatch 失敗時，把目前 resource 加入 attempted_resources
- 若 retry_count < MAX_DISPATCH_RETRY（預設 2），用 get_fallback_resource 選下一個重試
- 三個都失敗才 mark_failed，同時寫 events 表（resource_all_exhausted）
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from models import SessionLocal, Submission, SubmissionHistory

logger = logging.getLogger("modelhub.poller.queue_dispatcher")

DISPATCH_INTERVAL_SECONDS = int(os.environ.get("MODELHUB_DISPATCH_INTERVAL", "30"))
MAX_CONCURRENT_TRAININGS = int(os.environ.get("MODELHUB_MAX_CONCURRENT", "2"))
MAX_DISPATCH_RETRY = int(os.environ.get("MODELHUB_MAX_DISPATCH_RETRY", "2"))

_scheduler = None
_last_dispatch_at: Optional[datetime] = None


def get_last_dispatch_at() -> Optional[datetime]:
    return _last_dispatch_at


def _append_history(db, req_no: str, action: str, note: Optional[str] = None,
                    meta: Optional[dict] = None) -> None:
    import json
    row = SubmissionHistory(
        req_no=req_no,
        action=action,
        actor="queue-dispatcher",
        note=note,
        meta=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(row)


def dispatch_next() -> dict:
    """
    主派發邏輯（同步，供 APScheduler 呼叫）：
    1. 若 running >= MAX_CONCURRENT_TRAININGS，跳過
    2. 取最高優先序最舊的 waiting 條目
    3. mark_dispatching → 執行訓練派發
    4. 根據結果 mark_running 或 mark_failed
    """
    global _last_dispatch_at
    _last_dispatch_at = datetime.utcnow()

    from queue_manager import QueueManager

    db = SessionLocal()
    try:
        running_count = QueueManager.count_running(db)
        if running_count >= MAX_CONCURRENT_TRAININGS:
            logger.debug(
                "dispatch_next: running=%d >= max=%d, skip",
                running_count, MAX_CONCURRENT_TRAININGS,
            )
            return {"skipped": True, "reason": f"running={running_count} >= max={MAX_CONCURRENT_TRAININGS}"}

        entry = QueueManager.peek_next(db)
        if not entry:
            logger.debug("dispatch_next: queue empty")
            return {"skipped": True, "reason": "queue_empty"}

        req_no = entry.req_no
        entry_id = entry.id

        # 先標記 dispatching（佔位，防止並發多次派發同一條）
        QueueManager.mark_dispatching(db, entry_id)
        db.commit()

        logger.info(
            "dispatch_next: dispatching req=%s priority=%s entry_id=%d",
            req_no, entry.priority, entry_id,
        )

        # 執行實際訓練派發（含 M18 cross-resource retry）
        result = _do_dispatch_with_retry(req_no, db, entry_id)
        return result

    except Exception as e:
        logger.exception("dispatch_next exception: %s", e)
        return {"error": str(e)}
    finally:
        db.close()


def _do_dispatch(req_no: str, db) -> tuple[bool, str, str]:
    """
    實際派發訓練任務，與 actions._handle_start_training_resource_bg 邏輯一致。
    回傳 (success: bool, resource: str, reason: str)

    Race condition 防護（fix/queue-dispatcher-kernel-race）：
    在推進狀態前先確認資源可行性。若 Kaggle 為最佳資源但
    kaggle_kernel_slug 尚未 attach，回傳特殊 sentinel
    "PENDING_KERNEL" 讓呼叫端改設 pending_kernel 狀態，
    避免誤走 local fallback 或觸發 blocked。
    """
    import json as _json
    from datetime import datetime as _dt

    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        return False, "", f"submission {req_no} not found"
    if obj.status != "approved":
        return False, "", f"status={obj.status} (expected approved)"

    # --- Kernel attach 前置檢查（race condition fix）---
    # 若 ResourceProber 判定使用 Kaggle 但 kaggle_kernel_slug 尚未設定，
    # 不推進到 training，改回傳 PENDING_KERNEL sentinel，
    # 由 dispatch_next 負責標記 pending_kernel 並等待 attach-kernel 完成。
    try:
        from resources.prober import ResourceProber
        _prober = ResourceProber()
        _best = _prober.get_best_resource(db=db)
        _resource_hint = _best.get("resource", "local")
        if _resource_hint == "kaggle" and not obj.kaggle_kernel_slug:
            logger.info(
                "_do_dispatch: req=%s kaggle selected but kernel not yet attached, "
                "returning PENDING_KERNEL",
                req_no,
            )
            return False, "PENDING_KERNEL", "kaggle_kernel_slug not set"
    except Exception as _probe_err:
        logger.warning(
            "_do_dispatch: ResourceProber failed for req=%s (%s), proceeding with dispatch",
            req_no, _probe_err,
        )

    now = _dt.utcnow()
    obj.status = "training"
    obj.training_started_at = now
    obj.total_attempts = (obj.total_attempts or 0) + 1

    _append_history(db, req_no=req_no, action="start_training",
                    note="queue_dispatcher auto-triggered")
    db.commit()
    db.refresh(obj)

    # 呼叫資源派發
    try:
        from routers.actions import _handle_start_training_resource
        _handle_start_training_resource(obj, db)
        resource = obj.training_resource or "unknown"
        db.commit()
        return True, resource, ""
    except Exception as e:
        return False, "", str(e)


def _do_dispatch_with_retry(req_no: str, db, entry_id: int) -> dict:
    """
    M18-2: Cross-Resource Auto Retry 派發邏輯。

    1. 讀取 TrainingQueue entry 的 attempted_resources（JSON list）
    2. 呼叫 _do_dispatch()
    3. 若成功 → mark_running，回傳 dispatched
    4. 若 PENDING_KERNEL → mark_pending_kernel（不計入 retry）
    5. 若失敗：
       a. 把 failed resource 加入 attempted_resources
       b. 若 retry_count < MAX_DISPATCH_RETRY：
          - 用 get_fallback_resource(attempted) 找下一個
          - 重設 submission resource hint，mark_dispatching 繼續
       c. 若 retry 耗盡 → mark_failed + 寫 resource_all_exhausted event
    """
    from queue_manager import QueueManager
    from models import TrainingQueue

    entry = db.query(TrainingQueue).filter(TrainingQueue.id == entry_id).first()
    if not entry:
        return {"error": f"entry_id={entry_id} not found"}

    # 讀已嘗試的資源
    try:
        attempted = json.loads(entry.attempted_resources or "[]")
    except (ValueError, TypeError):
        attempted = []

    # 第一次嘗試
    success, resource, reason = _do_dispatch(req_no, db)

    if success:
        QueueManager.mark_running(db, entry_id, target_resource=resource)
        _append_history(db, req_no, "queue_dispatched",
                        note=f"訓練派發成功，resource={resource}",
                        meta={"entry_id": entry_id, "resource": resource})
        db.commit()
        logger.info("dispatch_next: req=%s dispatched to %s", req_no, resource)
        return {"dispatched": req_no, "resource": resource}

    if resource == "PENDING_KERNEL":
        QueueManager.mark_pending_kernel(db, entry_id)
        _append_history(db, req_no, "queue_pending_kernel",
                        note="Kaggle kernel 尚未 attach，等待 attach-kernel 後自動重試",
                        meta={"entry_id": entry_id})
        db.commit()
        logger.info("dispatch_next: req=%s pending_kernel", req_no)
        return {"pending_kernel": req_no, "reason": reason}

    # 失敗：記錄 attempted resource
    failed_resource = _extract_failed_resource(resource, reason)
    if failed_resource and failed_resource not in attempted:
        attempted.append(failed_resource)

    retry_count = entry.retry_count or 0
    logger.warning(
        "dispatch_next: req=%s dispatch failed resource=%s reason=%s retry_count=%d",
        req_no, failed_resource, reason, retry_count,
    )

    # M18-2: retry loop
    while retry_count < MAX_DISPATCH_RETRY:
        # 找下一個 fallback resource
        try:
            from resources.prober import ResourceProber
            prober = ResourceProber()
            fallback = prober.get_fallback_resource(attempted=attempted, db=db)
        except Exception as _pe:
            logger.warning("dispatch_retry: ResourceProber failed: %s", _pe)
            fallback = None

        if not fallback:
            logger.info(
                "dispatch_retry: req=%s no fallback available after %d retries, attempted=%s",
                req_no, retry_count + 1, attempted,
            )
            break

        fallback_resource = fallback.get("resource", "unknown")
        retry_count += 1

        # 更新 entry
        entry.attempted_resources = json.dumps(attempted, ensure_ascii=False)
        entry.retry_count = retry_count
        db.commit()

        _append_history(
            db, req_no, "queue_retry_fallback",
            note=f"fallback: {failed_resource}→{fallback_resource}（retry {retry_count}/{MAX_DISPATCH_RETRY}）",
            meta={
                "entry_id": entry_id,
                "from_resource": failed_resource,
                "to_resource": fallback_resource,
                "retry_count": retry_count,
                "attempted": attempted,
            },
        )
        db.commit()
        logger.info(
            "dispatch_retry: req=%s fallback: %s→%s retry=%d",
            req_no, failed_resource, fallback_resource, retry_count,
        )

        # 重試派發（目前邏輯：_do_dispatch 內部會 re-probe，fallback hint 透過 prober 自動選到）
        success, resource, reason = _do_dispatch(req_no, db)

        if success:
            QueueManager.mark_running(db, entry_id, target_resource=resource)
            _append_history(db, req_no, "queue_dispatched",
                            note=f"retry 派發成功，resource={resource}",
                            meta={"entry_id": entry_id, "resource": resource, "retry_count": retry_count})
            db.commit()
            logger.info("dispatch_retry: req=%s dispatched to %s (retry %d)", req_no, resource, retry_count)
            return {"dispatched": req_no, "resource": resource, "retry_count": retry_count}

        if resource == "PENDING_KERNEL":
            QueueManager.mark_pending_kernel(db, entry_id)
            db.commit()
            return {"pending_kernel": req_no, "reason": reason}

        # 繼續累積 attempted
        failed_resource = _extract_failed_resource(resource, reason)
        if failed_resource and failed_resource not in attempted:
            attempted.append(failed_resource)

    # 所有 retry 耗盡 → mark_failed
    final_reason = f"所有 resource 嘗試失敗（attempted={attempted}）: {reason}"
    QueueManager.mark_failed(db, entry_id, final_reason)
    _append_history(db, req_no, "queue_dispatch_failed",
                    note=f"Cross-resource retry 耗盡，派發失敗：{final_reason}",
                    meta={"entry_id": entry_id, "attempted": attempted, "reason": reason})
    db.commit()
    logger.warning("dispatch_next: req=%s all resources exhausted attempted=%s", req_no, attempted)

    # M18-4: 寫 resource_all_exhausted event（不走 CMC）
    try:
        from pollers.health_checker import record_resource_all_exhausted
        record_resource_all_exhausted(db, req_no=req_no, attempted=attempted)
    except Exception as _evt_err:
        logger.warning("dispatch_next: failed to write resource_all_exhausted event: %s", _evt_err)

    return {"failed": req_no, "reason": final_reason, "attempted": attempted}


def _extract_failed_resource(resource: str, reason: str) -> str:
    """
    從 _do_dispatch 回傳的 resource 字串提取標準 resource key。
    e.g. "kaggle" → "kaggle", "ssh@192.168.50.83" → "ssh@192.168.50.83"
    空字串或 unknown → "unknown"
    """
    if not resource or resource in ("", "unknown"):
        # 嘗試從 reason 猜
        reason_lower = (reason or "").lower()
        if "kaggle" in reason_lower:
            return "kaggle"
        if "lightning" in reason_lower:
            return "lightning"
        if "ssh" in reason_lower:
            return "ssh"
        return "unknown"
    return resource


def start_scheduler():
    global _scheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        logger.warning("apscheduler not installed, queue dispatcher disabled")
        return None

    if _scheduler:
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone="Asia/Taipei")
    _scheduler.add_job(
        dispatch_next, "interval",
        seconds=DISPATCH_INTERVAL_SECONDS,
        id="queue-dispatcher",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info("Queue dispatcher started (interval=%ds, max_concurrent=%d)",
                DISPATCH_INTERVAL_SECONDS, MAX_CONCURRENT_TRAININGS)
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Queue dispatcher stopped")
