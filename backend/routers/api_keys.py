"""
routers/api_keys.py — Sprint 7.1 API Key 管理（LIDS token 限定）
"""
import secrets
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import ApiKey, get_db
from auth import CurrentUser, require_role

router = APIRouter()


class ApiKeyCreate(BaseModel):
    name: str


class ApiKeyOut(BaseModel):
    id: int
    key: str
    name: str
    created_by: Optional[str]
    created_at: datetime
    last_used_at: Optional[datetime]
    disabled: bool

    model_config = {"from_attributes": True}


class ApiKeyListItem(BaseModel):
    """列表用，隱藏 key 大部分，只留 prefix 給識別"""
    id: int
    key_preview: str
    name: str
    created_by: Optional[str]
    created_at: datetime
    last_used_at: Optional[datetime]
    disabled: bool


@router.get("/", response_model=List[ApiKeyListItem])
async def list_api_keys(
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
    _role: dict = require_role("admin"),
):
    rows = db.query(ApiKey).order_by(ApiKey.id.desc()).all()
    return [
        ApiKeyListItem(
            id=r.id,
            key_preview=f"{r.key[:6]}...{r.key[-4:]}" if len(r.key) > 12 else r.key,
            name=r.name,
            created_by=r.created_by,
            created_at=r.created_at,
            last_used_at=r.last_used_at,
            disabled=r.disabled,
        )
        for r in rows
    ]


@router.post("/", response_model=ApiKeyOut, status_code=201)
async def create_api_key(
    payload: ApiKeyCreate,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
    _role: dict = require_role("admin"),
):
    if not payload.name.strip():
        raise HTTPException(status_code=422, detail="name 不可為空")
    token = secrets.token_urlsafe(32)
    actor = (current_user or {}).get("preferred_username") \
        or (current_user or {}).get("email") or "unknown"
    row = ApiKey(
        key=token,
        name=payload.name.strip(),
        created_by=actor,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.post("/{key_id}/disable")
async def disable_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
    _role: dict = require_role("admin"),
):
    row = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="ApiKey not found")
    row.disabled = True
    db.commit()
    return {"id": row.id, "disabled": True}


@router.post("/{key_id}/enable")
async def enable_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
    _role: dict = require_role("admin"),
):
    row = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="ApiKey not found")
    row.disabled = False
    db.commit()
    return {"id": row.id, "disabled": False}
