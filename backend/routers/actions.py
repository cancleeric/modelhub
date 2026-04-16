"""
routers/actions.py — 需求單狀態機 API

狀態轉移圖：
  draft → submitted → approved → training → trained → accepted
                    ↘ rejected                      ↘ failed
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import Submission, get_db
from notifications import notify
from auth import CurrentUser

router = APIRouter()

# 合法的狀態轉移 map：{action: (required_current_status, next_status)}
TRANSITIONS: dict[str, tuple[str, str]] = {
    "submit":            ("draft",     "submitted"),
    "approve":           ("submitted", "approved"),
    "reject":            ("submitted", "rejected"),
    "start_training":    ("approved",  "training"),
    "complete_training": ("training",  "trained"),
    "accept":            ("trained",   "accepted"),
    "fail":              ("trained",   "failed"),
}


class ActionPayload(BaseModel):
    note: Optional[str] = None          # reviewer_note / acceptance_note
    actor: Optional[str] = None         # 操作人


@router.post("/{req_no}/actions/{action}")
async def perform_action(
    req_no: str,
    action: str,
    payload: ActionPayload = ActionPayload(),
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    if action not in TRANSITIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown action '{action}'. Valid actions: {list(TRANSITIONS.keys())}",
        )

    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")

    required_status, next_status = TRANSITIONS[action]
    if obj.status != required_status:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Action '{action}' requires status '{required_status}', "
                f"but current status is '{obj.status}'."
            ),
        )

    # 執行狀態轉移
    obj.status = next_status

    # 附加欄位更新
    now = datetime.utcnow()
    if action in ("approve", "reject"):
        obj.reviewed_by = payload.actor
        obj.reviewed_at = now
        if payload.note:
            obj.reviewer_note = payload.note

    db.commit()
    db.refresh(obj)

    # 非同步通知（fire-and-forget，失敗不影響 API 回應）
    await _send_notification(action, obj, payload)

    return {
        "req_no": req_no,
        "action": action,
        "status": obj.status,
        "message": f"Transition '{action}' applied successfully.",
    }


async def _send_notification(action: str, obj: Submission, payload: ActionPayload):
    """觸發對應通知，靜默失敗。"""
    try:
        if action == "submit":
            await notify(
                to="cto@hurricanecore.internal",
                message=(
                    f"[ModelHub] 新需求單待審：{obj.req_no} — "
                    f"{obj.req_name or obj.product}（提交人：{obj.submitter or '未知'}）"
                ),
            )
        elif action == "approve":
            if obj.submitter:
                await notify(
                    to=obj.submitter,
                    message=(
                        f"[ModelHub] 需求單 {obj.req_no} 已核准，可進入訓練階段。"
                        + (f"\n審核意見：{obj.reviewer_note}" if obj.reviewer_note else "")
                    ),
                )
        elif action == "reject":
            if obj.submitter:
                await notify(
                    to=obj.submitter,
                    message=(
                        f"[ModelHub] 需求單 {obj.req_no} 已拒絕。"
                        + (f"\n審核意見：{obj.reviewer_note}" if obj.reviewer_note else "")
                    ),
                )
        elif action == "accept":
            if obj.submitter:
                await notify(
                    to=obj.submitter,
                    message=f"[ModelHub] 需求單 {obj.req_no} 模型驗收通過。",
                )
        elif action == "fail":
            if obj.submitter:
                await notify(
                    to=obj.submitter,
                    message=f"[ModelHub] 需求單 {obj.req_no} 模型驗收未通過，請確認後續處理。",
                )
    except Exception:
        # 通知失敗不影響主流程
        pass
