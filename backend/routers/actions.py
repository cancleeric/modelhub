"""
routers/actions.py — 需求單狀態機 API

狀態轉移圖：
  draft → submitted → approved → training → trained → accepted
                    ↘ rejected ↗ (resubmit)             ↘ failed
"""

import json
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import Submission, SubmissionHistory, get_db
from notifications import notify_event
from auth import CurrentUser, CurrentUserOrApiKey, require_role

router = APIRouter()

# 合法的狀態轉移 map：{action: (required_current_status, next_status)}
TRANSITIONS: dict[str, tuple[str, str]] = {
    "submit":            ("draft",     "submitted"),
    "approve":           ("submitted", "approved"),
    "reject":            ("submitted", "rejected"),
    "resubmit":          ("rejected",  "submitted"),
    "start_training":    ("approved",  "training"),
    "complete_training": ("training",  "trained"),
    "accept":            ("trained",   "accepted"),
    "fail":              ("trained",   "failed"),
    "retrain":           ("failed",    "approved"),
}


def _record_history(
    db: Session,
    req_no: str,
    action: str,
    actor: Optional[str] = None,
    reasons: Optional[List[str]] = None,
    note: Optional[str] = None,
    meta: Optional[dict] = None,
) -> None:
    history = SubmissionHistory(
        req_no=req_no,
        action=action,
        actor=actor,
        reasons=json.dumps(reasons, ensure_ascii=False) if reasons else None,
        note=note,
        meta=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(history)


class ActionPayload(BaseModel):
    note: Optional[str] = None
    actor: Optional[str] = None


class RejectPayload(BaseModel):
    reasons: List[str]
    note: Optional[str] = None
    actor: Optional[str] = None


class ResubmitPayload(BaseModel):
    note: Optional[str] = None
    actor: Optional[str] = None


class HistoryOut(BaseModel):
    id: int
    req_no: str
    action: str
    actor: Optional[str]
    reasons: Optional[List[str]]
    note: Optional[str]
    meta: Optional[dict]
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# 結構化退件 / 補件（Sprint 2）
# ---------------------------------------------------------------------------

@router.post("/{req_no}/reject")
async def reject_submission(
    req_no: str,
    payload: RejectPayload,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
    _role: dict = require_role("reviewer"),
):
    """結構化退件：reasons=[...]、note=說明"""
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    if obj.status != "submitted":
        raise HTTPException(
            status_code=422,
            detail=f"reject requires status 'submitted', current='{obj.status}'",
        )
    if not payload.reasons:
        raise HTTPException(status_code=422, detail="reasons 不可為空")

    actor = payload.actor or (current_user or {}).get("preferred_username") or "unknown"
    now = datetime.utcnow()
    obj.status = "rejected"
    obj.rejection_reasons = json.dumps(payload.reasons, ensure_ascii=False)
    obj.rejection_note = payload.note
    obj.reviewed_by = actor
    obj.reviewed_at = now
    if payload.note:
        obj.reviewer_note = payload.note

    _record_history(
        db, req_no=req_no, action="reject", actor=actor,
        reasons=payload.reasons, note=payload.note,
    )
    db.commit()
    db.refresh(obj)
    await notify_event("reject", obj, actor=actor, note=payload.note)
    return {
        "req_no": req_no,
        "action": "reject",
        "status": obj.status,
        "rejection_reasons": payload.reasons,
    }


@router.post("/{req_no}/resubmit")
async def resubmit_submission(
    req_no: str,
    payload: ResubmitPayload,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """補件 resubmit：rejected → submitted，resubmit_count +1"""
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    if obj.status != "rejected":
        raise HTTPException(
            status_code=422,
            detail=f"resubmit requires status 'rejected', current='{obj.status}'",
        )

    actor = payload.actor or (current_user or {}).get("preferred_username") or "unknown"
    now = datetime.utcnow()
    obj.status = "submitted"
    obj.resubmit_count = (obj.resubmit_count or 0) + 1
    obj.resubmitted_at = now

    _record_history(
        db, req_no=req_no, action="resubmit", actor=actor,
        note=payload.note,
        meta={"resubmit_count": obj.resubmit_count},
    )
    db.commit()
    db.refresh(obj)
    await notify_event("resubmit", obj, actor=actor, note=payload.note)
    return {
        "req_no": req_no,
        "action": "resubmit",
        "status": obj.status,
        "resubmit_count": obj.resubmit_count,
    }


@router.get("/{req_no}/history", response_model=List[HistoryOut])
async def get_submission_history(
    req_no: str,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """審核軌跡（照時間遞增排）"""
    rows = (
        db.query(SubmissionHistory)
        .filter(SubmissionHistory.req_no == req_no)
        .order_by(SubmissionHistory.created_at.asc())
        .all()
    )
    result = []
    for row in rows:
        result.append(HistoryOut(
            id=row.id,
            req_no=row.req_no,
            action=row.action,
            actor=row.actor,
            reasons=json.loads(row.reasons) if row.reasons else None,
            note=row.note,
            meta=json.loads(row.meta) if row.meta else None,
            created_at=row.created_at,
        ))
    return result


# ---------------------------------------------------------------------------
# 通用狀態機 action（其他 transition）
# ---------------------------------------------------------------------------

@router.post("/{req_no}/actions/{action}")
async def perform_action(
    req_no: str,
    action: str,
    payload: ActionPayload = ActionPayload(),
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    if action not in TRANSITIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown action '{action}'. Valid: {list(TRANSITIONS.keys())}",
        )

    # P3-3: approve 需要 reviewer role
    if action == "approve":
        import os as _os
        skip = _os.getenv("SKIP_ROLE_CHECK", "false").lower() == "true"
        if not skip:
            _role_claim = _os.getenv("MODELHUB_ROLE_CLAIM", "modelhub_role")
            user_role = (current_user or {}).get(_role_claim)
            # API Key 使用者（sub 以 api_key: 開頭）略過
            is_api_key = str((current_user or {}).get("sub", "")).startswith("api_key:")
            if not is_api_key and user_role != "reviewer":
                raise HTTPException(
                    status_code=403,
                    detail=f"approve requires role 'reviewer', current='{user_role}'",
                )

    # reject / resubmit 有獨立端點用於結構化欄位，但保留這裡作為相容後門
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")

    required_status, next_status = TRANSITIONS[action]
    if obj.status != required_status:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Action '{action}' requires status '{required_status}', "
                f"current='{obj.status}'."
            ),
        )

    actor = payload.actor or (current_user or {}).get("preferred_username") or "unknown"
    obj.status = next_status
    now = datetime.utcnow()

    if action in ("approve", "reject"):
        obj.reviewed_by = actor
        obj.reviewed_at = now
        if payload.note:
            obj.reviewer_note = payload.note
    if action == "start_training":
        obj.training_started_at = now
        obj.total_attempts = (obj.total_attempts or 0) + 1
    if action == "complete_training":
        obj.training_completed_at = now
    if action == "resubmit":
        obj.resubmit_count = (obj.resubmit_count or 0) + 1
        obj.resubmitted_at = now

    _record_history(
        db, req_no=req_no, action=action, actor=actor, note=payload.note,
    )
    db.commit()
    db.refresh(obj)

    await notify_event(action, obj, actor=actor, note=payload.note)

    return {
        "req_no": req_no,
        "action": action,
        "status": obj.status,
    }
