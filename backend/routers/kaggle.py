"""
routers/kaggle.py — Kaggle Kernel 整合 API（Sprint 3）
"""
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import Submission, get_db
from auth import CurrentUserOrApiKey
from pollers.kaggle_poller import _fetch_kernel_status, _get_kaggle_api, poll_once

router = APIRouter()


class AttachKernelPayload(BaseModel):
    slug: str                 # e.g. boardgamegroup/mh-2026-001
    version: Optional[int] = None
    actor: Optional[str] = None


@router.post("/{req_no}/attach-kernel")
async def attach_kernel(
    req_no: str,
    payload: AttachKernelPayload,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    if obj.status not in ("approved", "training"):
        raise HTTPException(
            status_code=422,
            detail=f"attach-kernel requires status approved/training, current='{obj.status}'",
        )

    now = datetime.utcnow()
    obj.kaggle_kernel_slug = payload.slug
    obj.kaggle_kernel_version = payload.version
    obj.kaggle_status = "queued"
    obj.kaggle_status_updated_at = now
    if obj.status == "approved":
        obj.status = "training"
        obj.training_started_at = now
        obj.total_attempts = (obj.total_attempts or 0) + 1

    # append history
    from models import SubmissionHistory
    db.add(SubmissionHistory(
        req_no=req_no,
        action="attach_kernel",
        actor=payload.actor or "unknown",
        meta=json.dumps({"slug": payload.slug, "version": payload.version}, ensure_ascii=False),
    ))
    db.commit()
    db.refresh(obj)
    return {
        "req_no": req_no,
        "kaggle_kernel_slug": obj.kaggle_kernel_slug,
        "kaggle_kernel_version": obj.kaggle_kernel_version,
        "status": obj.status,
        "kaggle_status": obj.kaggle_status,
    }


@router.get("/{req_no}/kaggle-status")
async def get_kaggle_status(
    req_no: str,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    return {
        "req_no": req_no,
        "kaggle_kernel_slug": obj.kaggle_kernel_slug,
        "kaggle_kernel_version": obj.kaggle_kernel_version,
        "kaggle_status": obj.kaggle_status,
        "kaggle_status_updated_at": obj.kaggle_status_updated_at,
        "kaggle_log_url": obj.kaggle_log_url,
        "training_started_at": obj.training_started_at,
        "training_completed_at": obj.training_completed_at,
        "gpu_seconds": obj.gpu_seconds,
        "estimated_cost_usd": obj.estimated_cost_usd,
    }


@router.post("/{req_no}/refresh-kaggle")
async def refresh_kaggle(
    req_no: str,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """手動強制 poll 一次（只處理此 submission）"""
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    if not obj.kaggle_kernel_slug:
        raise HTTPException(status_code=422, detail="此工單尚未綁定 kaggle kernel")

    api = _get_kaggle_api()
    res = await _fetch_kernel_status(api, obj.kaggle_kernel_slug)
    if not res:
        return {"ok": False, "detail": "無法取得 kaggle status（可能 CLI 未安裝或 env 未設）"}

    # 讓 full poller 走一次完整流程（含 complete 處理）
    summary = await poll_once()
    db.refresh(obj)
    return {
        "ok": True,
        "fetched": res,
        "poll_summary": summary,
        "kaggle_status": obj.kaggle_status,
        "status": obj.status,
    }


@router.post("/{req_no}/refresh-lightning")
async def refresh_lightning(
    req_no: str,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    手動觸發 Lightning poller 掃描此工單狀態。
    適用於 training_resource=lightning 的 submission。
    """
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    if obj.training_resource != "lightning":
        raise HTTPException(
            status_code=422,
            detail=f"此工單 training_resource='{obj.training_resource}'，非 lightning",
        )

    studio_name = obj.lightning_studio_name or obj.kaggle_kernel_slug
    if not studio_name:
        raise HTTPException(status_code=422, detail="此工單尚未綁定 Lightning Studio")

    from pollers.lightning_poller import poll_once as lightning_poll_once
    summary = lightning_poll_once()
    db.refresh(obj)
    return {
        "ok": True,
        "poll_summary": summary,
        "studio_name": studio_name,
        "kaggle_status": obj.kaggle_status,
        "status": obj.status,
        "lightning_studio_name": obj.lightning_studio_name,
    }
