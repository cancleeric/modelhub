"""
queue_manager.py — 持久化訓練隊列管理器（Sprint 20）

所有操作皆以傳入的 db Session 執行，呼叫端負責 commit 時機。
優先序排序：P0 > P1 > P2 > P3（字典序恰好與優先序一致）。
同一優先序內以 enqueued_at ASC 排序（先進先出）。
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from models import TrainingQueue

logger = logging.getLogger("modelhub.queue_manager")

# 優先序 mapping（數字越小越優先）
_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _priority_key(priority: str) -> int:
    return _PRIORITY_ORDER.get(priority, 99)


class QueueManager:
    """訓練隊列管理器（stateless，所有方法為 class method）"""

    @staticmethod
    def enqueue(db: Session, req_no: str, priority: str = "P2") -> TrainingQueue:
        """
        將 req_no 加入訓練隊列。
        若已存在相同 req_no（任意狀態），先刪除再重新入隊，確保冪等。
        """
        # 冪等：若已在隊列中，移除再重新入隊
        existing = db.query(TrainingQueue).filter(TrainingQueue.req_no == req_no).first()
        if existing:
            logger.info("enqueue: req=%s already in queue (status=%s), re-enqueue", req_no, existing.status)
            db.delete(existing)
            db.flush()

        entry = TrainingQueue(
            req_no=req_no,
            priority=priority,
            status="waiting",
            enqueued_at=datetime.utcnow(),
            retry_count=0,
        )
        db.add(entry)
        db.flush()
        db.refresh(entry)
        logger.info("enqueue: req=%s priority=%s entry_id=%d", req_no, priority, entry.id)
        return entry

    @staticmethod
    def peek_next(db: Session) -> Optional[TrainingQueue]:
        """
        取優先序最高 + 最舊的 waiting 條目（不 pop）。
        P0 > P1 > P2 > P3，同優先序以 enqueued_at ASC。
        """
        rows = (
            db.query(TrainingQueue)
            .filter(TrainingQueue.status == "waiting")
            .all()
        )
        if not rows:
            return None

        rows.sort(key=lambda e: (_priority_key(e.priority), e.enqueued_at))
        return rows[0]

    @staticmethod
    def mark_dispatching(db: Session, entry_id: int) -> None:
        """將條目標記為 dispatching，設 dispatched_at"""
        entry = db.query(TrainingQueue).filter(TrainingQueue.id == entry_id).first()
        if not entry:
            logger.warning("mark_dispatching: entry_id=%d not found", entry_id)
            return
        entry.status = "dispatching"
        entry.dispatched_at = datetime.utcnow()
        db.flush()

    @staticmethod
    def mark_running(db: Session, entry_id: int, target_resource: Optional[str] = None) -> None:
        """將條目標記為 running"""
        entry = db.query(TrainingQueue).filter(TrainingQueue.id == entry_id).first()
        if not entry:
            logger.warning("mark_running: entry_id=%d not found", entry_id)
            return
        entry.status = "running"
        if target_resource:
            entry.target_resource = target_resource
        db.flush()

    @staticmethod
    def mark_done(db: Session, entry_id: int) -> None:
        """將條目標記為 done"""
        entry = db.query(TrainingQueue).filter(TrainingQueue.id == entry_id).first()
        if not entry:
            logger.warning("mark_done: entry_id=%d not found", entry_id)
            return
        entry.status = "done"
        db.flush()

    @staticmethod
    def mark_failed(db: Session, entry_id: int, reason: str) -> None:
        """將條目標記為 failed，記錄 error_reason"""
        entry = db.query(TrainingQueue).filter(TrainingQueue.id == entry_id).first()
        if not entry:
            logger.warning("mark_failed: entry_id=%d not found", entry_id)
            return
        entry.status = "failed"
        entry.error_reason = reason[:500] if reason else reason
        db.flush()

    @staticmethod
    def mark_done_by_req(db: Session, req_no: str) -> None:
        """依 req_no 標記完成（供 poller 呼叫）"""
        entry = db.query(TrainingQueue).filter(TrainingQueue.req_no == req_no).first()
        if entry and entry.status in ("running", "dispatching"):
            entry.status = "done"
            db.flush()

    @staticmethod
    def mark_failed_by_req(db: Session, req_no: str, reason: str) -> None:
        """依 req_no 標記失敗（供 poller 呼叫）"""
        entry = db.query(TrainingQueue).filter(TrainingQueue.req_no == req_no).first()
        if entry and entry.status in ("running", "dispatching"):
            entry.status = "failed"
            entry.error_reason = reason[:500] if reason else reason
            db.flush()

    @staticmethod
    def mark_pending_kernel(db: Session, entry_id: int) -> None:
        """
        將條目標記為 pending_kernel。
        發生於 queue_dispatcher 偵測到 kaggle_kernel_slug 尚未 attach 時，
        避免誤判無 GPU 資源而進入 blocked。
        attach-kernel 完成後需呼叫 reset_pending_kernel 恢復為 waiting。
        """
        entry = db.query(TrainingQueue).filter(TrainingQueue.id == entry_id).first()
        if not entry:
            logger.warning("mark_pending_kernel: entry_id=%d not found", entry_id)
            return
        entry.status = "pending_kernel"
        db.flush()

    @staticmethod
    def reset_pending_kernel(db: Session, req_no: str) -> bool:
        """
        將 pending_kernel 狀態的條目重設為 waiting，讓 dispatcher 下次重試。
        由 attach-kernel endpoint 在成功 attach 後呼叫。
        回傳 True 表示有條目被重設，False 表示無對應 pending_kernel 條目。
        """
        entry = (
            db.query(TrainingQueue)
            .filter(TrainingQueue.req_no == req_no, TrainingQueue.status == "pending_kernel")
            .first()
        )
        if not entry:
            return False
        entry.status = "waiting"
        db.flush()
        logger.info("reset_pending_kernel: req=%s entry_id=%d → waiting", req_no, entry.id)
        return True

    @staticmethod
    def count_running(db: Session) -> int:
        """回傳當前 running + dispatching 條目數量"""
        return (
            db.query(TrainingQueue)
            .filter(TrainingQueue.status.in_(["running", "dispatching"]))
            .count()
        )

    @staticmethod
    def get_queue_position(db: Session, req_no: str) -> int:
        """
        回傳指定 req_no 在 waiting 隊列中的排名（從 1 開始）。
        若不在 waiting 狀態，回傳 -1。
        """
        rows = (
            db.query(TrainingQueue)
            .filter(TrainingQueue.status == "waiting")
            .all()
        )
        rows.sort(key=lambda e: (_priority_key(e.priority), e.enqueued_at))
        for idx, entry in enumerate(rows, start=1):
            if entry.req_no == req_no:
                return idx
        return -1

    @staticmethod
    def get_all_waiting(db: Session) -> list[TrainingQueue]:
        """取全部 waiting 條目，依優先序排序"""
        rows = (
            db.query(TrainingQueue)
            .filter(TrainingQueue.status == "waiting")
            .all()
        )
        rows.sort(key=lambda e: (_priority_key(e.priority), e.enqueued_at))
        return rows

    @staticmethod
    def get_all_running(db: Session) -> list[TrainingQueue]:
        """取全部 running + dispatching 條目"""
        return (
            db.query(TrainingQueue)
            .filter(TrainingQueue.status.in_(["running", "dispatching"]))
            .all()
        )
