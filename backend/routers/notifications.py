"""
routers/notifications.py — M22 Phase 4 通知 API

Endpoints:
  GET  /api/notifications                  — 列當前 user 未讀/全部通知
  POST /api/notifications/mark-read        — 批次標已讀

Tenant 隔離：
  只回傳 recipient_email == current_user email 的通知。
  不跨 tenant 查看。
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import CurrentUserOrApiKey
from models import CommentNotification, SubmissionComment, get_db
from routers.comments import _get_user_email

_logger = logging.getLogger("modelhub.routers.notifications")

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class NotificationOut(BaseModel):
    id: int
    comment_id: int
    recipient_email: str
    type: str           # mention / reply / new_comment
    req_no: str
    read_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


class MarkReadPayload(BaseModel):
    ids: Optional[List[int]] = None   # None / [] = 標全部


# ---------------------------------------------------------------------------
# GET /api/notifications
# ---------------------------------------------------------------------------

@router.get("/notifications", response_model=List[NotificationOut])
async def list_notifications(
    unread_only: bool = Query(default=False, description="只回傳未讀"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    列出當前使用者的通知（unread first）。
    - unread_only=true：只回傳 read_at IS NULL
    - 預設：未讀優先，再依 created_at DESC
    """
    email = _get_user_email(current_user)
    if not email or email == "unknown":
        raise HTTPException(status_code=401, detail="Cannot determine user identity")

    query = (
        db.query(CommentNotification)
        .filter(CommentNotification.recipient_email == email)
    )
    if unread_only:
        query = query.filter(CommentNotification.read_at.is_(None))

    # 未讀優先（read_at IS NULL 排前面），再 created_at DESC
    from sqlalchemy import case, desc, nulls_first
    try:
        query = query.order_by(
            nulls_first(CommentNotification.read_at),
            desc(CommentNotification.created_at),
        )
    except Exception:
        # SQLite 可能不支援 nulls_first，fallback
        query = query.order_by(
            CommentNotification.read_at.asc(),
            CommentNotification.created_at.desc(),
        )

    offset = (page - 1) * page_size
    rows = query.offset(offset).limit(page_size).all()

    result = []
    for row in rows:
        # 取 req_no from comment
        comment = db.query(SubmissionComment).filter(
            SubmissionComment.id == row.comment_id
        ).first()
        req_no = comment.req_no if comment else ""
        result.append(
            NotificationOut(
                id=row.id,
                comment_id=row.comment_id,
                recipient_email=row.recipient_email,
                type=row.type,
                req_no=req_no,
                read_at=row.read_at,
                created_at=row.created_at,
            )
        )
    return result


# ---------------------------------------------------------------------------
# POST /api/notifications/mark-read
# ---------------------------------------------------------------------------

@router.post("/notifications/mark-read")
async def mark_notifications_read(
    payload: MarkReadPayload,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    標記通知為已讀。
    - payload.ids=[1,2,3]：標指定 id
    - payload.ids=None 或 []：標全部未讀
    """
    email = _get_user_email(current_user)
    if not email or email == "unknown":
        raise HTTPException(status_code=401, detail="Cannot determine user identity")

    now = datetime.utcnow()
    query = db.query(CommentNotification).filter(
        CommentNotification.recipient_email == email,
        CommentNotification.read_at.is_(None),
    )

    if payload.ids:
        query = query.filter(CommentNotification.id.in_(payload.ids))

    updated = query.all()
    count = len(updated)
    for n in updated:
        n.read_at = now

    db.commit()
    return {"marked_read": count}
