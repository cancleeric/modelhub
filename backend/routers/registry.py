from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import ModelVersion, get_db

router = APIRouter()


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
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/", response_model=List[ModelVersionOut])
def list_versions(
    req_no: Optional[str] = None,
    product: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
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
def get_version(id: int, db: Session = Depends(get_db)):
    obj = db.query(ModelVersion).filter(ModelVersion.id == id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="ModelVersion not found")
    return obj


@router.post("/", response_model=ModelVersionOut, status_code=201)
def create_version(payload: ModelVersionCreate, db: Session = Depends(get_db)):
    obj = ModelVersion(**payload.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.patch("/{id}", response_model=ModelVersionOut)
def update_version(id: int, payload: ModelVersionUpdate, db: Session = Depends(get_db)):
    obj = db.query(ModelVersion).filter(ModelVersion.id == id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="ModelVersion not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(obj, field, value)
    db.commit()
    db.refresh(obj)
    return obj


@router.delete("/{id}", status_code=204)
def delete_version(id: int, db: Session = Depends(get_db)):
    obj = db.query(ModelVersion).filter(ModelVersion.id == id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="ModelVersion not found")
    db.delete(obj)
    db.commit()
