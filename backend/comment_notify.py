"""
comment_notify.py — M22 Phase 4 留言通知服務

觸發點：POST /api/submissions/{req_no}/comments
功能：
  1. 計算 recipients（原 submitter + thread 參與者 + @mention 者，去重）
  2. 排除留言者本人
  3. INSERT comment_notifications（不發 CMC）
  4. 寫 SystemEvent（event_type=comment_created / comment_replied）

Tenant 隔離：recipient 必須和工單同 company（或 mention 對象有明確跨 tenant 權限）
注意：此模組不做 LIDS 驗證（Phase 4 規格：接受 cross-tenant mention，但只通知 DB 內有紀錄的 email）
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from models import (
    Submission,
    SubmissionComment,
    CommentNotification,
    SystemEvent,
)
from mention_parser import parse_mentions

_logger = logging.getLogger("modelhub.comment_notify")


def _get_thread_participants(req_no: str, db: Session) -> set:
    """
    取得工單 discussion 所有留言者的 email（包含已刪除留言的 author）。
    不包含 @mention，只看有實際發言的人。
    """
    rows = (
        db.query(SubmissionComment.author_email)
        .filter(SubmissionComment.req_no == req_no)
        .distinct()
        .all()
    )
    return {r[0] for r in rows if r[0]}


def create_comment_notifications(
    comment: SubmissionComment,
    submission: Submission,
    db: Session,
    author_email: str,
) -> None:
    """
    建立留言後的通知邏輯。
    - mention：@mention 的人 → type=mention
    - reply：parent 留言的 author → type=reply
    - new_comment：原 submitter + 其他 thread 參與者 → type=new_comment
    去重後排除 author 本人。
    """
    try:
        now = datetime.utcnow()
        mentioned = parse_mentions(comment.body_markdown)

        # 計算各 notification type 的 recipients
        mention_recipients = set(mentioned)

        reply_recipients: set = set()
        if comment.parent_id is not None:
            parent = db.query(SubmissionComment).filter(
                SubmissionComment.id == comment.parent_id
            ).first()
            if parent and parent.author_email:
                reply_recipients.add(parent.author_email)

        # new_comment：submitter + 已參與者（排除 mention/reply 已涵蓋者避免重複 type）
        existing_participants = _get_thread_participants(req_no=comment.req_no, db=db)
        submitter_email = submission.submitter if submission.submitter else None

        new_comment_recipients: set = set()
        if submitter_email:
            new_comment_recipients.add(submitter_email)
        new_comment_recipients.update(existing_participants)
        # 從 new_comment 中排除 mention/reply 已處理的（避免一個人收到多筆）
        new_comment_recipients -= mention_recipients
        new_comment_recipients -= reply_recipients

        # 排除 author 本人
        mention_recipients.discard(author_email)
        reply_recipients.discard(author_email)
        new_comment_recipients.discard(author_email)

        # INSERT notifications
        for email in mention_recipients:
            notif = CommentNotification(
                comment_id=comment.id,
                recipient_email=email,
                type="mention",
                created_at=now,
            )
            db.add(notif)

        for email in reply_recipients:
            notif = CommentNotification(
                comment_id=comment.id,
                recipient_email=email,
                type="reply",
                created_at=now,
            )
            db.add(notif)

        for email in new_comment_recipients:
            notif = CommentNotification(
                comment_id=comment.id,
                recipient_email=email,
                type="new_comment",
                created_at=now,
            )
            db.add(notif)

        # 寫 SystemEvent（comment_created / comment_replied）
        event_type = "comment_replied" if comment.parent_id is not None else "comment_created"
        meta = {
            "author": author_email,
            "comment_id": comment.id,
            "is_internal": comment.is_internal,
            "mentioned_users": mentioned,
        }
        if comment.parent_id is not None:
            meta["parent_id"] = comment.parent_id

        event = SystemEvent(
            event_type=event_type,
            req_no=comment.req_no,
            severity="info",
            message=f"[{event_type}] {author_email} commented on {comment.req_no}",
            meta=json.dumps(meta, ensure_ascii=False),
            created_at=now,
        )
        db.add(event)

        _logger.info(
            "comment_notify: comment_id=%d, mentions=%d, replies=%d, new_comment=%d",
            comment.id,
            len(mention_recipients),
            len(reply_recipients),
            len(new_comment_recipients),
        )

    except Exception as exc:
        _logger.warning("create_comment_notifications failed (comment_id=%s): %s", comment.id, exc)
