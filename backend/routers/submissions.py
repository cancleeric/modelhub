from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from models import Submission, get_db
from auth import CurrentUser, CurrentUserOrApiKey

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SubmissionCreate(BaseModel):
    req_name: Optional[str] = None
    product: str
    company: str
    submitter: Optional[str] = None
    purpose: Optional[str] = None
    priority: str = "P2"
    model_type: Optional[str] = None
    class_list: Optional[str] = None
    map50_threshold: Optional[float] = None   # 舊欄位相容
    map50_target: Optional[float] = None
    map50_95_target: Optional[float] = None
    inference_latency_ms: Optional[int] = None
    model_size_limit_mb: Optional[int] = None
    arch: Optional[str] = "yolov8m"
    input_spec: Optional[str] = None
    deploy_env: Optional[str] = None
    dataset_source: Optional[str] = None
    dataset_count: Optional[str] = None
    dataset_val_count: Optional[int] = None
    dataset_test_count: Optional[int] = None
    class_count: Optional[int] = None
    label_format: Optional[str] = None
    kaggle_dataset_url: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_train_count: Optional[int] = None
    expected_delivery: Optional[str] = None
    # Sprint 6
    max_retries: Optional[int] = 2
    max_budget_usd: Optional[float] = 5.0


class SubmissionUpdate(BaseModel):
    req_name: Optional[str] = None
    product: Optional[str] = None
    company: Optional[str] = None
    submitter: Optional[str] = None
    purpose: Optional[str] = None
    priority: Optional[str] = None
    model_type: Optional[str] = None
    class_list: Optional[str] = None
    map50_threshold: Optional[float] = None
    map50_target: Optional[float] = None
    map50_95_target: Optional[float] = None
    inference_latency_ms: Optional[int] = None
    model_size_limit_mb: Optional[int] = None
    arch: Optional[str] = None
    input_spec: Optional[str] = None
    deploy_env: Optional[str] = None
    dataset_source: Optional[str] = None
    dataset_count: Optional[str] = None
    dataset_val_count: Optional[int] = None
    dataset_test_count: Optional[int] = None
    class_count: Optional[int] = None
    label_format: Optional[str] = None
    kaggle_dataset_url: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_train_count: Optional[int] = None
    expected_delivery: Optional[str] = None
    status: Optional[str] = None
    reviewer_note: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class SubmissionOut(BaseModel):
    id: int
    req_no: str
    req_name: Optional[str]
    product: str
    company: str
    submitter: Optional[str]
    purpose: Optional[str]
    priority: str
    model_type: Optional[str]
    class_list: Optional[str]
    map50_threshold: Optional[float]
    map50_target: Optional[float]
    map50_95_target: Optional[float]
    inference_latency_ms: Optional[int]
    model_size_limit_mb: Optional[int]
    arch: Optional[str]
    input_spec: Optional[str]
    deploy_env: Optional[str]
    dataset_source: Optional[str]
    dataset_count: Optional[str]
    dataset_val_count: Optional[int]
    dataset_test_count: Optional[int]
    class_count: Optional[int]
    label_format: Optional[str]
    kaggle_dataset_url: Optional[str]
    dataset_path: Optional[str]
    dataset_train_count: Optional[int]
    expected_delivery: Optional[str]
    status: str
    reviewer_note: Optional[str]
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]
    # Sprint 2
    rejection_reasons: Optional[str] = None
    rejection_note: Optional[str] = None
    resubmit_count: Optional[int] = 0
    resubmitted_at: Optional[datetime] = None
    # Sprint 3
    kaggle_kernel_slug: Optional[str] = None
    kaggle_kernel_version: Optional[int] = None
    kaggle_status: Optional[str] = None
    kaggle_status_updated_at: Optional[datetime] = None
    kaggle_log_url: Optional[str] = None
    training_started_at: Optional[datetime] = None
    training_completed_at: Optional[datetime] = None
    # Sprint 4
    gpu_seconds: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    total_attempts: Optional[int] = 0
    # Sprint 6
    max_retries: Optional[int] = 2
    retry_count: Optional[int] = 0
    max_budget_usd: Optional[float] = 5.0
    budget_exceeded_notified: Optional[bool] = False
    # Dataset unblock
    dataset_status: Optional[str] = "ready"
    blocked_reason: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SubmissionCreateResult(BaseModel):
    submission: SubmissionOut
    warnings: list[str] = []
    suggestions: list[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/stats/summary")
async def get_stats_summary(
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    """各狀態件數統計"""
    rows = (
        db.query(Submission.status, func.count(Submission.id).label("count"))
        .group_by(Submission.status)
        .all()
    )
    total = sum(r.count for r in rows)
    by_status = {r.status: r.count for r in rows}
    return {"total": total, "by_status": by_status}


@router.get("/", response_model=List[SubmissionOut])
async def list_submissions(
    status: Optional[str] = None,
    product: Optional[str] = None,
    dataset_status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    q = db.query(Submission)
    if status:
        q = q.filter(Submission.status == status)
    if product:
        q = q.filter(Submission.product == product)
    if dataset_status:
        q = q.filter(Submission.dataset_status == dataset_status)
    return q.order_by(Submission.created_at.desc()).all()


@router.get("/{req_no}", response_model=SubmissionOut)
async def get_submission(
    req_no: str,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    return obj


@router.post("/", response_model=SubmissionCreateResult, status_code=201)
async def create_submission(
    payload: SubmissionCreate,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """建立需求單，req_no 自動生成。回傳 warnings（rule-based）+ suggestions（LLM）。"""
    from validators import validate_submission
    from advisors.llm_advisor import review_submission
    import asyncio as _asyncio
    warnings_task = validate_submission(payload)
    suggestions_task = review_submission(payload)
    warnings, suggestions = await _asyncio.gather(warnings_task, suggestions_task)

    year = datetime.utcnow().year
    prefix = f"MH-{year}-"
    latest = (
        db.query(Submission)
        .filter(Submission.req_no.like(f"{prefix}%"))
        .order_by(Submission.req_no.desc())
        .first()
    )
    if latest:
        try:
            last_seq = int(latest.req_no.split("-")[-1])
        except ValueError:
            last_seq = 0
    else:
        last_seq = 0
    new_seq = last_seq + 1
    req_no = f"{prefix}{new_seq:03d}"

    obj = Submission(**payload.model_dump(), req_no=req_no)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return SubmissionCreateResult(
        submission=SubmissionOut.model_validate(obj),
        warnings=warnings,
        suggestions=suggestions,
    )


@router.patch("/{req_no}", response_model=SubmissionOut)
async def update_submission(
    req_no: str,
    payload: SubmissionUpdate,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(obj, field, value)
    db.commit()
    db.refresh(obj)
    return obj


@router.delete("/{req_no}", status_code=204)
async def delete_submission(
    req_no: str,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    db.delete(obj)
    db.commit()
