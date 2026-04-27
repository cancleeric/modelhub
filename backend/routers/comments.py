"""
routers/comments.py — M22 Discussion 留言系統

Endpoints:
  GET    /api/submissions/{req_no}/comments
  POST   /api/submissions/{req_no}/comments
  PATCH  /api/comments/{id}
  DELETE /api/comments/{id}

權限矩陣：
  - is_internal comment：only reviewer / superadmin / CTO 看得到
  - 只有 author 或 superadmin 可 PATCH / DELETE
  - tenant 隔離：req_no 必須屬於 current_user 的 company（或 reviewer+）

注意：tenant 驗證走 Submission.company 查詢，
      reviewer / superadmin 有 cross-tenant 存取權。
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import CurrentUserOrApiKey, _ROLE_CLAIM_KEY, _SKIP_ROLE_CHECK
from models import (
    Submission,
    SubmissionComment,
    SubmissionAttachment,
    SubmissionHistory,
    get_db,
)
from mention_parser import parse_mentions

_logger = logging.getLogger("modelhub.routers.comments")

# 編輯窗口（分鐘）— 超過此時限 body 標記為「已編輯」但仍允許（不阻擋）
_EDIT_WINDOW_MINUTES = int(os.getenv("COMMENT_EDIT_WINDOW_MINUTES", "5"))

# 高權限 role（可看 internal comment、可刪任意留言）
_PRIVILEGED_ROLES = {"reviewer", "cto", "superadmin"}

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class AttachmentOut(BaseModel):
    id: int
    filename: str
    size_bytes: int
    mime_type: str
    uploaded_by: str
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class CommentOut(BaseModel):
    id: int
    req_no: str
    author_email: str
    body_markdown: str
    is_internal: bool
    parent_id: Optional[int]
    created_at: datetime
    updated_at: Optional[datetime]
    deleted_at: Optional[datetime]
    attachments: List[AttachmentOut] = []
    replies: List["CommentOut"] = []
    mentioned_users: List[str] = []   # M22 Phase 4：@mention 解析結果

    model_config = {"from_attributes": True}


# 允許 self-referential forward ref
CommentOut.model_rebuild()


class CommentCreatePayload(BaseModel):
    body_markdown: str
    is_internal: bool = False
    parent_id: Optional[int] = None
    attachment_ids: List[int] = []


class CommentEditPayload(BaseModel):
    body_markdown: str


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_user_role(current_user: dict) -> str:
    """從 userinfo 取出 role。API Key 使用者視為 reviewer（機器帳號）。"""
    if not current_user:
        return "submitter"
    sub = str(current_user.get("sub", ""))
    if sub.startswith("api_key:"):
        return "reviewer"
    return current_user.get(_ROLE_CLAIM_KEY) or "submitter"


def _is_privileged(current_user: dict) -> bool:
    return _SKIP_ROLE_CHECK or _get_user_role(current_user) in _PRIVILEGED_ROLES


def _get_user_email(current_user: dict) -> str:
    if not current_user:
        return "unknown"
    return (
        current_user.get("email")
        or current_user.get("preferred_username")
        or current_user.get("sub")
        or "unknown"
    )


def _assert_submission_access(
    req_no: str,
    db: Session,
    current_user: dict,
) -> Submission:
    """
    取得 submission，並確認 tenant 存取權限。
    - 特權 role → cross-tenant OK
    - 一般使用者 → company 需一致
    """
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail=f"Submission {req_no} not found")

    if _is_privileged(current_user):
        return obj

    # 一般使用者：tenant 隔離（email domain 或 preferred_username 比對 company）
    user_company = current_user.get("company") or current_user.get("tenant")
    if user_company and obj.company and user_company.lower() != obj.company.lower():
        raise HTTPException(status_code=403, detail="Tenant mismatch: access denied")

    return obj


def _build_comment_out(comment: SubmissionComment, include_internal: bool) -> Optional[CommentOut]:
    """
    組裝 CommentOut，含附件與 replies（1 層）。
    若 include_internal=False 且 is_internal=True，回傳 None（呼叫端跳過）。
    """
    if not include_internal and comment.is_internal:
        return None

    attachments = [
        AttachmentOut(
            id=a.id,
            filename=a.filename,
            size_bytes=a.size_bytes,
            mime_type=a.mime_type,
            uploaded_by=a.uploaded_by,
            uploaded_at=a.uploaded_at,
        )
        for a in (comment.attachments or [])
    ]

    # 只組裝直接 replies（1 層，parent_id == comment.id）
    children = []
    for reply in comment.replies.filter(SubmissionComment.parent_id == comment.id).order_by(
        SubmissionComment.created_at.asc()
    ).all():
        if include_internal or not reply.is_internal:
            r_attachments = [
                AttachmentOut(
                    id=a.id,
                    filename=a.filename,
                    size_bytes=a.size_bytes,
                    mime_type=a.mime_type,
                    uploaded_by=a.uploaded_by,
                    uploaded_at=a.uploaded_at,
                )
                for a in (reply.attachments or [])
            ]
            children.append(
                CommentOut(
                    id=reply.id,
                    req_no=reply.req_no,
                    author_email=reply.author_email,
                    body_markdown=reply.body_markdown,
                    is_internal=reply.is_internal,
                    parent_id=reply.parent_id,
                    created_at=reply.created_at,
                    updated_at=reply.updated_at,
                    deleted_at=reply.deleted_at,
                    attachments=r_attachments,
                    replies=[],
                )
            )

    return CommentOut(
        id=comment.id,
        req_no=comment.req_no,
        author_email=comment.author_email,
        body_markdown=comment.body_markdown,
        is_internal=comment.is_internal,
        parent_id=comment.parent_id,
        created_at=comment.created_at,
        updated_at=comment.updated_at,
        deleted_at=comment.deleted_at,
        attachments=attachments,
        replies=children,
        mentioned_users=parse_mentions(comment.body_markdown),
    )


def _update_submission_discussion_stats(req_no: str, db: Session) -> None:
    """更新 submissions.discussion_count 與 last_activity_at。"""
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        return
    # flush 確保 pending 的 deleted_at 寫入，count query 才能看到
    db.flush()
    count = (
        db.query(SubmissionComment)
        .filter(
            SubmissionComment.req_no == req_no,
            SubmissionComment.deleted_at.is_(None),
        )
        .count()
    )
    last = (
        db.query(SubmissionComment)
        .filter(SubmissionComment.req_no == req_no)
        .order_by(SubmissionComment.created_at.desc())
        .first()
    )
    obj.discussion_count = count
    obj.last_activity_at = last.created_at if last else None


def _record_history(db: Session, req_no: str, action: str, actor: str, note: Optional[str] = None, meta: Optional[dict] = None) -> None:
    history = SubmissionHistory(
        req_no=req_no,
        action=action,
        actor=actor,
        note=note,
        meta=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(history)


# ---------------------------------------------------------------------------
# GET /api/submissions/{req_no}/comments
# ---------------------------------------------------------------------------

@router.get("/submissions/{req_no}/comments", response_model=List[CommentOut])
async def list_comments(
    req_no: str,
    include_internal: bool = Query(default=False),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """列出工單的所有頂層留言（含 replies）。"""
    _assert_submission_access(req_no, db, current_user)

    # 非特權使用者不得帶 include_internal=true
    if include_internal and not _is_privileged(current_user):
        raise HTTPException(status_code=403, detail="Requires reviewer+ role to view internal comments")

    offset = (page - 1) * page_size

    # 只取頂層 comment（parent_id IS NULL）
    top_level = (
        db.query(SubmissionComment)
        .filter(
            SubmissionComment.req_no == req_no,
            SubmissionComment.parent_id.is_(None),
        )
        .order_by(SubmissionComment.created_at.asc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    result = []
    for c in top_level:
        out = _build_comment_out(c, include_internal)
        if out is not None:
            result.append(out)

    return result


# ---------------------------------------------------------------------------
# POST /api/submissions/{req_no}/comments
# ---------------------------------------------------------------------------

@router.post("/submissions/{req_no}/comments", response_model=CommentOut, status_code=201)
async def create_comment(
    req_no: str,
    payload: CommentCreatePayload,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """建立留言（含 reply）。"""
    _assert_submission_access(req_no, db, current_user)

    # is_internal 只有特權使用者可以設定
    if payload.is_internal and not _is_privileged(current_user):
        raise HTTPException(status_code=403, detail="Requires reviewer+ role to post internal comment")

    # 驗證 parent_id：只允許 1 層 reply（parent 必須是頂層）
    if payload.parent_id is not None:
        parent = db.query(SubmissionComment).filter(
            SubmissionComment.id == payload.parent_id,
            SubmissionComment.req_no == req_no,
        ).first()
        if not parent:
            raise HTTPException(status_code=404, detail="Parent comment not found")
        if parent.parent_id is not None:
            raise HTTPException(status_code=422, detail="Only 1-level reply is supported")

    author = _get_user_email(current_user)
    now = datetime.utcnow()

    comment = SubmissionComment(
        req_no=req_no,
        author_email=author,
        body_markdown=payload.body_markdown,
        is_internal=payload.is_internal,
        parent_id=payload.parent_id,
        created_at=now,
    )
    db.add(comment)
    db.flush()  # 取得 comment.id

    # 關聯附件
    if payload.attachment_ids:
        for att_id in payload.attachment_ids:
            att = db.query(SubmissionAttachment).filter(
                SubmissionAttachment.id == att_id,
                SubmissionAttachment.req_no == req_no,
                SubmissionAttachment.comment_id.is_(None),  # 尚未被其他 comment 使用
            ).first()
            if att:
                att.comment_id = comment.id

    _update_submission_discussion_stats(req_no, db)
    _record_history(
        db, req_no=req_no, action="comment_created",
        actor=author,
        note=f"comment_id={comment.id}",
        meta={"comment_id": comment.id, "is_internal": payload.is_internal},
    )

    # M22 Phase 4: 通知（mention / reply / new_comment）
    try:
        from comment_notify import create_comment_notifications
        sub = db.query(Submission).filter(Submission.req_no == req_no).first()
        create_comment_notifications(
            comment=comment,
            submission=sub,
            db=db,
            author_email=author,
        )
    except Exception as _e:
        _logger.warning("notification creation failed: %s", _e)

    db.commit()
    db.refresh(comment)

    out = _build_comment_out(comment, include_internal=True)
    return out


# ---------------------------------------------------------------------------
# PATCH /api/comments/{id}
# ---------------------------------------------------------------------------

@router.patch("/comments/{comment_id}", response_model=CommentOut)
async def edit_comment(
    comment_id: int,
    payload: CommentEditPayload,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """編輯留言（只有 author 或 superadmin 可改）。"""
    comment = db.query(SubmissionComment).filter(SubmissionComment.id == comment_id).first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    author = _get_user_email(current_user)
    user_role = _get_user_role(current_user)

    if author != comment.author_email and user_role != "superadmin" and not _SKIP_ROLE_CHECK:
        raise HTTPException(status_code=403, detail="Only the author or superadmin can edit this comment")

    # 驗證 submission 存取（tenant 隔離）
    _assert_submission_access(comment.req_no, db, current_user)

    now = datetime.utcnow()
    comment.body_markdown = payload.body_markdown
    comment.updated_at = now

    db.commit()
    db.refresh(comment)

    out = _build_comment_out(comment, include_internal=True)
    return out


# ---------------------------------------------------------------------------
# DELETE /api/comments/{id}
# ---------------------------------------------------------------------------

@router.delete("/comments/{comment_id}", status_code=204)
async def delete_comment(
    comment_id: int,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """Soft delete 留言（deleted_at = now，body 替換為刪除提示）。"""
    comment = db.query(SubmissionComment).filter(SubmissionComment.id == comment_id).first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    author = _get_user_email(current_user)
    user_role = _get_user_role(current_user)

    if author != comment.author_email and user_role != "superadmin" and not _SKIP_ROLE_CHECK:
        raise HTTPException(status_code=403, detail="Only the author or superadmin can delete this comment")

    # 驗證 submission 存取（tenant 隔離）
    _assert_submission_access(comment.req_no, db, current_user)

    now = datetime.utcnow()
    comment.deleted_at = now
    comment.body_markdown = "此留言已刪除"

    _update_submission_discussion_stats(comment.req_no, db)
    db.commit()
    return None


# ---------------------------------------------------------------------------
# 內部函式：供 reject API 呼叫，建立第一筆 comment
# ---------------------------------------------------------------------------

def create_reject_comment(
    req_no: str,
    reasons: List[str],
    note: Optional[str],
    actor_email: str,
    db: Session,
) -> int:
    """
    退件時自動建立首筆 Discussion comment。
    回傳 comment.id。
    """
    # 將 reasons 組成 markdown checklist + note
    lines = ["**退件原因：**", ""]
    for reason in reasons:
        lines.append(f"- [ ] {reason}")
    if note:
        lines.append("")
        lines.append("**補充說明：**")
        lines.append("")
        lines.append(note)

    body = "\n".join(lines)
    now = datetime.utcnow()

    comment = SubmissionComment(
        req_no=req_no,
        author_email=actor_email,
        body_markdown=body,
        is_internal=False,
        parent_id=None,
        created_at=now,
    )
    db.add(comment)
    db.flush()

    _update_submission_discussion_stats(req_no, db)
    _record_history(
        db, req_no=req_no, action="reject_comment_created",
        actor=actor_email,
        note=f"auto-created reject comment, comment_id={comment.id}",
        meta={"comment_id": comment.id},
    )

    return comment.id


# ---------------------------------------------------------------------------
# GET /api/comments/search — M22 Phase 4 全文搜尋
# ---------------------------------------------------------------------------

class CommentSearchOut(BaseModel):
    id: int
    req_no: str
    author_email: str
    body_markdown: str
    is_internal: bool
    parent_id: Optional[int]
    created_at: datetime
    mentioned_users: List[str] = []

    model_config = {"from_attributes": True}


@router.get("/comments/search", response_model=List[CommentSearchOut])
async def search_comments(
    q: str = Query(..., min_length=1, description="搜尋關鍵字"),
    req_no: Optional[str] = Query(default=None, description="限定工單（可省略）"),
    author: Optional[str] = Query(default=None, description="限定 author email"),
    since: Optional[str] = Query(default=None, description="起始時間 ISO8601（UTC）"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    搜尋留言。使用 SQLite LIKE / PostgreSQL ilike。
    - q：必填，body_markdown 模糊比對
    - req_no：可選，限定工單
    - author：可選，限定留言者 email
    - since：可選，ISO8601 UTC 時間，只回傳 created_at >= since 的結果
    - tenant 隔離：非特權使用者只能搜自己公司的工單留言
    """
    include_internal = _is_privileged(current_user)

    query = db.query(SubmissionComment).filter(
        SubmissionComment.body_markdown.ilike(f"%{q}%"),
        SubmissionComment.deleted_at.is_(None),
    )

    if not include_internal:
        query = query.filter(SubmissionComment.is_internal.is_(False))

    if req_no:
        # 同時做 tenant 驗證
        _assert_submission_access(req_no, db, current_user)
        query = query.filter(SubmissionComment.req_no == req_no)
    elif not _is_privileged(current_user):
        # 一般使用者只能搜自己公司的工單留言
        user_company = current_user.get("company") or current_user.get("tenant")
        if user_company:
            # join submissions 過濾 company
            query = query.join(
                Submission,
                Submission.req_no == SubmissionComment.req_no,
            ).filter(Submission.company == user_company)

    if author:
        query = query.filter(SubmissionComment.author_email.ilike(f"%{author}%"))

    if since:
        try:
            since_dt = datetime.fromisoformat(since.rstrip("Z"))
            query = query.filter(SubmissionComment.created_at >= since_dt)
        except ValueError:
            raise HTTPException(status_code=422, detail="since 格式錯誤，需為 ISO8601 UTC（e.g. 2026-01-01T00:00:00）")

    offset = (page - 1) * page_size
    rows = (
        query
        .order_by(SubmissionComment.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return [
        CommentSearchOut(
            id=r.id,
            req_no=r.req_no,
            author_email=r.author_email,
            body_markdown=r.body_markdown,
            is_internal=r.is_internal,
            parent_id=r.parent_id,
            created_at=r.created_at,
            mentioned_users=parse_mentions(r.body_markdown),
        )
        for r in rows
    ]
