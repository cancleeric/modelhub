from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import Submission, get_db

router = APIRouter()


class SubmissionCreate(BaseModel):
    req_no: str
    product: str
    company: str
    submitter: Optional[str] = None
    purpose: Optional[str] = None
    class_list: Optional[str] = None
    map50_threshold: Optional[float] = None
    input_spec: Optional[str] = None
    deploy_env: Optional[str] = None
    dataset_source: Optional[str] = None
    dataset_count: Optional[str] = None
    label_format: Optional[str] = None
    expected_delivery: Optional[str] = None
    status: str = "pending"


class SubmissionUpdate(BaseModel):
    product: Optional[str] = None
    company: Optional[str] = None
    submitter: Optional[str] = None
    purpose: Optional[str] = None
    class_list: Optional[str] = None
    map50_threshold: Optional[float] = None
    input_spec: Optional[str] = None
    deploy_env: Optional[str] = None
    dataset_source: Optional[str] = None
    dataset_count: Optional[str] = None
    label_format: Optional[str] = None
    expected_delivery: Optional[str] = None
    status: Optional[str] = None


class SubmissionOut(BaseModel):
    id: int
    req_no: str
    product: str
    company: str
    submitter: Optional[str]
    purpose: Optional[str]
    class_list: Optional[str]
    map50_threshold: Optional[float]
    input_spec: Optional[str]
    deploy_env: Optional[str]
    dataset_source: Optional[str]
    dataset_count: Optional[str]
    label_format: Optional[str]
    expected_delivery: Optional[str]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/", response_model=List[SubmissionOut])
def list_submissions(
    status: Optional[str] = None,
    product: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Submission)
    if status:
        q = q.filter(Submission.status == status)
    if product:
        q = q.filter(Submission.product == product)
    return q.order_by(Submission.created_at.desc()).all()


@router.get("/{req_no}", response_model=SubmissionOut)
def get_submission(req_no: str, db: Session = Depends(get_db)):
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    return obj


@router.post("/", response_model=SubmissionOut, status_code=201)
def create_submission(payload: SubmissionCreate, db: Session = Depends(get_db)):
    existing = db.query(Submission).filter(Submission.req_no == payload.req_no).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"req_no {payload.req_no} already exists")
    obj = Submission(**payload.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.patch("/{req_no}", response_model=SubmissionOut)
def update_submission(req_no: str, payload: SubmissionUpdate, db: Session = Depends(get_db)):
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(obj, field, value)
    db.commit()
    db.refresh(obj)
    return obj


@router.delete("/{req_no}", status_code=204)
def delete_submission(req_no: str, db: Session = Depends(get_db)):
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    db.delete(obj)
    db.commit()
