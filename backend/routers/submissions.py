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


class TrainingResultUpdate(BaseModel):
    """Sprint 8.2 — 訓練腳本回寫用 schema"""
    status: str                                    # "trained" | "training_failed"
    metrics: Optional[dict] = None                # {"map50": 0.62, "epochs": 20, ...}
    model_path: Optional[str] = None
    notes: Optional[str] = None
    per_class_metrics: Optional[dict] = None      # Sprint 13 P2-A: {class_name: ap50}


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
    reviewer_note: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    model_output_path: Optional[str] = None


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
    model_output_path: Optional[str] = None
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
    # Sprint 15 P2-3 / Sprint 16 P1-2
    training_resource: Optional[str] = None
    # Sprint 13 P2-A
    per_class_metrics: Optional[str] = None
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


@router.patch("/{req_no}/training-result", response_model=SubmissionOut)
async def update_training_result(
    req_no: str,
    payload: TrainingResultUpdate,
    db: Session = Depends(get_db),
    _user: dict = CurrentUserOrApiKey,
):
    """
    Sprint 8.2 — 訓練腳本完成後自動回寫狀態。

    允許 status: trained | training_failed
    來源狀態必須是：training、training_failed（重送）、approved（自動補 start）
    同時更新 training_completed_at, kaggle_status, blocked_reason（如需要）
    Sprint 13 P0-B: 加入狀態機前置檢查
    Sprint 13 P2-C: training_failed 時自動寫 retrain_suggested history
    Sprint 13 P2-D: 同步更新 ModelVersion.pass_fail
    Sprint 13 P2-A: 儲存 per_class_metrics
    """
    valid_statuses = {"trained", "training_failed"}
    if payload.status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"status 必須是 {valid_statuses}，收到 {payload.status!r}",
        )

    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Sprint 13 P0-B: 狀態機前置檢查
    allowed_source_statuses = {"training", "training_failed", "approved"}
    if obj.status not in allowed_source_statuses:
        raise HTTPException(
            status_code=400,
            detail=(
                f"需求單 {req_no} 目前狀態為 {obj.status!r}，"
                f"不允許直接更新訓練結果。"
                f"允許的來源狀態：{sorted(allowed_source_statuses)}"
            ),
        )

    now = datetime.utcnow()

    # 若來源狀態為 approved，自動補 training transition
    if obj.status == "approved":
        obj.status = "training"
        obj.training_started_at = now
        if obj.total_attempts is None:
            obj.total_attempts = 0
        obj.total_attempts += 1

    obj.status = payload.status
    obj.training_completed_at = now

    if payload.metrics:
        # map50_threshold 是需求人填的門檻規格，不能被訓練結果覆蓋
        # 實測值僅用於 pass/fail 比對，不寫回 map50_threshold
        pass

    # Sprint 13 P2-A: 儲存 per_class_metrics
    if payload.per_class_metrics:
        import json as _json
        try:
            obj.per_class_metrics = _json.dumps(payload.per_class_metrics, ensure_ascii=False)
        except Exception:
            pass

    if payload.model_path:
        obj.model_output_path = payload.model_path  # Sprint 10: 寫入專屬欄位
        # N3: 同步寫入對應 ModelVersion.file_path 及 pass_fail
        try:
            from models import ModelVersion
            mv = (
                db.query(ModelVersion)
                .filter(ModelVersion.req_no == req_no)
                .order_by(ModelVersion.id.desc())
                .first()
            )
            if mv:
                if not mv.file_path:
                    mv.file_path = payload.model_path
                # Sprint 13 P2-D: 同步更新 pass_fail
                map50_val = None
                if payload.metrics and "map50" in payload.metrics:
                    try:
                        map50_val = float(payload.metrics["map50"])
                    except Exception:
                        pass
                if map50_val is not None and obj.map50_target is not None:
                    mv.pass_fail = "pass" if map50_val >= obj.map50_target else "fail"
                elif map50_val is not None and obj.map50_threshold is not None:
                    mv.pass_fail = "pass" if map50_val >= obj.map50_threshold else "fail"
        except Exception:
            pass

    if payload.notes:
        obj.reviewer_note = (obj.reviewer_note or "") + f"\n[auto] {payload.notes}"

    # 若是 training_failed 且沒有 blocked_reason，補上
    if payload.status == "training_failed" and not obj.blocked_reason:
        note_text = payload.notes or "訓練失敗"
        obj.blocked_reason = f"[sprint8.2] {note_text}"

    # Sprint 13 P2-C: training_failed 且達最大重試次數 → 寫 retrain_suggested history
    if payload.status == "training_failed":
        retry_count = obj.retry_count or 0
        max_retries = obj.max_retries if obj.max_retries is not None else 2
        if retry_count >= max_retries:
            from models import SubmissionHistory
            from notifications import notify_event
            import json as _json_hist
            map50_val_hist = None
            if payload.metrics and "map50" in payload.metrics:
                try:
                    map50_val_hist = float(payload.metrics["map50"])
                except Exception:
                    pass
            map50_target_hist = obj.map50_target or obj.map50_threshold
            gap_desc = ""
            if map50_val_hist is not None and map50_target_hist is not None:
                gap = map50_target_hist - map50_val_hist
                gap_desc = f"mAP50={map50_val_hist:.4f}，與 baseline 差距 {gap:+.4f}"
            else:
                gap_desc = payload.notes or "詳見訓練 log"
            hist_note = f"訓練失敗達最大重試次數（{max_retries}）：{gap_desc}"
            hist = SubmissionHistory(
                req_no=req_no,
                action="retrain_suggested",
                actor="modelhub_auto",
                note=hist_note,
                meta=_json_hist.dumps({
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                    "map50": map50_val_hist,
                    "map50_target": map50_target_hist,
                }, ensure_ascii=False),
            )
            db.add(hist)
            import asyncio as _asyncio
            try:
                _asyncio.create_task(notify_event("retrain_suggested", obj, note=hist_note))
            except RuntimeError:
                pass  # 不在 event loop 中時跳過，不阻斷

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
