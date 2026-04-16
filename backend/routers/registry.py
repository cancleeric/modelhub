from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import ModelVersion, Submission, get_db
from auth import CurrentUser

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ModelVersionCreate(BaseModel):
    req_no: str
    product: str
    model_name: str
    version: str
    train_date: Optional[str] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    file_path: Optional[str] = None
    status: str = "active"
    notes: Optional[str] = None
    kaggle_kernel_url: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    arch: Optional[str] = None
    map50_actual: Optional[float] = None
    map50_95_actual: Optional[float] = None
    pass_fail: Optional[str] = None
    accepted_by: Optional[str] = None
    accepted_at: Optional[datetime] = None
    acceptance_note: Optional[str] = None


class ModelVersionUpdate(BaseModel):
    product: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[str] = None
    train_date: Optional[str] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    file_path: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    kaggle_kernel_url: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    arch: Optional[str] = None
    map50_actual: Optional[float] = None
    map50_95_actual: Optional[float] = None
    pass_fail: Optional[str] = None
    accepted_by: Optional[str] = None
    accepted_at: Optional[datetime] = None
    acceptance_note: Optional[str] = None


class ModelVersionOut(BaseModel):
    id: int
    req_no: str
    product: str
    model_name: str
    version: str
    train_date: Optional[str]
    map50: Optional[float]
    map50_95: Optional[float]
    file_path: Optional[str]
    status: str
    notes: Optional[str]
    kaggle_kernel_url: Optional[str]
    epochs: Optional[int]
    batch_size: Optional[int]
    arch: Optional[str]
    map50_actual: Optional[float]
    map50_95_actual: Optional[float]
    pass_fail: Optional[str]
    accepted_by: Optional[str]
    accepted_at: Optional[datetime]
    acceptance_note: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class AcceptancePayload(BaseModel):
    map50_actual: float
    map50_95_actual: Optional[float] = None
    acceptance_note: Optional[str] = None
    accepted_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/by-req/{req_no}", response_model=List[ModelVersionOut])
async def list_versions_by_req(
    req_no: str,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    """單一需求單的所有版本"""
    return (
        db.query(ModelVersion)
        .filter(ModelVersion.req_no == req_no)
        .order_by(ModelVersion.created_at.desc())
        .all()
    )


@router.get("/", response_model=List[ModelVersionOut])
async def list_versions(
    req_no: Optional[str] = None,
    product: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    q = db.query(ModelVersion)
    if req_no:
        q = q.filter(ModelVersion.req_no == req_no)
    if product:
        q = q.filter(ModelVersion.product == product)
    if status:
        q = q.filter(ModelVersion.status == status)
    return q.order_by(ModelVersion.created_at.desc()).all()


@router.get("/{id}", response_model=ModelVersionOut)
async def get_version(
    id: int,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    obj = db.query(ModelVersion).filter(ModelVersion.id == id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="ModelVersion not found")
    return obj


@router.post("/", response_model=ModelVersionOut, status_code=201)
async def create_version(
    payload: ModelVersionCreate,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    obj = ModelVersion(**payload.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.patch("/{id}", response_model=ModelVersionOut)
async def update_version(
    id: int,
    payload: ModelVersionUpdate,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    obj = db.query(ModelVersion).filter(ModelVersion.id == id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="ModelVersion not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(obj, field, value)
    db.commit()
    db.refresh(obj)
    return obj


@router.post("/{id}/accept", response_model=ModelVersionOut)
async def accept_version(
    id: int,
    payload: AcceptancePayload,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    """
    驗收模型版本：記錄 map50_actual、pass_fail，並與對應需求單的 map50_target 比對。
    """
    obj = db.query(ModelVersion).filter(ModelVersion.id == id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="ModelVersion not found")

    obj.map50_actual = payload.map50_actual
    obj.map50_95_actual = payload.map50_95_actual
    obj.acceptance_note = payload.acceptance_note
    obj.accepted_by = payload.accepted_by
    obj.accepted_at = datetime.utcnow()

    # 自動判定 pass/fail（依據需求單 map50_target）
    submission = db.query(Submission).filter(Submission.req_no == obj.req_no).first()
    target = submission.map50_target if submission else None
    if target is not None:
        obj.pass_fail = "pass" if payload.map50_actual >= target else "fail"
    else:
        obj.pass_fail = "pass"  # 無設定目標，預設通過

    db.commit()
    db.refresh(obj)
    return obj


@router.delete("/{id}", status_code=204)
async def delete_version(
    id: int,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    obj = db.query(ModelVersion).filter(ModelVersion.id == id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="ModelVersion not found")
    db.delete(obj)
    db.commit()
