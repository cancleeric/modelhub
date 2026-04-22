"""
routers/queue.py — 訓練隊列狀態 API（Sprint 20 Task 20-5）
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

import os

from models import get_db
from auth import CurrentUserOrApiKey

router = APIRouter()

# P2-30: 統一讀環境變數，與 queue_dispatcher 保持一致
MAX_CONCURRENT_TRAININGS = int(os.environ.get("MODELHUB_MAX_CONCURRENT", "2"))


class QueueWaitingItem(BaseModel):
    req_no: str
    priority: str
    position: int
    enqueued_at: datetime


class QueueRunningItem(BaseModel):
    req_no: str
    target_resource: Optional[str]
    dispatched_at: Optional[datetime]
    status: str


class QueueStatusResponse(BaseModel):
    waiting: List[QueueWaitingItem]
    running: List[QueueRunningItem]
    max_concurrent: int
    count_waiting: int
    count_running: int


@router.get("/status", response_model=QueueStatusResponse)
def queue_status(
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    回傳訓練隊列當前狀態：waiting 清單、running 清單、上限等。
    """
    from queue_manager import QueueManager

    waiting_entries = QueueManager.get_all_waiting(db)
    running_entries = QueueManager.get_all_running(db)

    waiting = [
        QueueWaitingItem(
            req_no=e.req_no,
            priority=e.priority,
            position=idx + 1,
            enqueued_at=e.enqueued_at,
        )
        for idx, e in enumerate(waiting_entries)
    ]

    running = [
        QueueRunningItem(
            req_no=e.req_no,
            target_resource=e.target_resource,
            dispatched_at=e.dispatched_at,
            status=e.status,
        )
        for e in running_entries
    ]

    return QueueStatusResponse(
        waiting=waiting,
        running=running,
        max_concurrent=MAX_CONCURRENT_TRAININGS,
        count_waiting=len(waiting),
        count_running=len(running),
    )
