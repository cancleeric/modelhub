"""External Model Registry — 外部 pretrained model 查詢 API.

提供：
  GET /api/external-models/{product}/{name}/path
      - 回傳 model 的 bind-mount 絕對路徑（供 Aegis 等 consumer 查詢）
      - 同步更新 model_versions.last_used_at
      - 使用 API Key 認證（機器對機器）

  GET /api/external-models/
      - 列出所有外部 model（status='registered'，有 external_source）
      - Bearer token 認證
"""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import CurrentUser, get_api_key
from models import ModelVersion, get_db

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ExternalModelPathResponse(BaseModel):
    req_no: str
    product: str
    model_name: str
    version: str
    file_path: str
    external_source: Optional[str]
    external_sha256: Optional[str]
    size_bytes: Optional[int]
    last_used_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ExternalModelOut(BaseModel):
    id: int
    req_no: str
    product: str
    model_name: str
    version: str
    arch: Optional[str]
    status: str
    file_path: Optional[str]
    external_source: Optional[str]
    external_sha256: Optional[str]
    size_bytes: Optional[int]
    last_used_at: Optional[datetime]
    notes: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{product}/{name}/path", response_model=ExternalModelPathResponse)
async def get_external_model_path(
    product: str,
    name: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key),
):
    """
    取得外部 model 的本機路徑。
    - product: 產品識別（例：aegis）
    - name: model 名稱（例：llama-guard-1b）
    - 回傳 file_path（bind-mount 路徑）供 consumer 載入 model
    - 同步更新 last_used_at
    """
    # name 允許用 - 或 _ 連接（llama-guard-1b / llama_guard_1b 都接受）
    name_normalized = name.lower().replace("_", "-")

    version = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.product == product,
            ModelVersion.status == "registered",
            ModelVersion.is_current == True,  # noqa: E712
        )
        .all()
    )

    # 找 model_name 能匹配的版本（normalize 比對）
    matched = None
    for v in version:
        if v.model_name.lower().replace("_", "-") == name_normalized:
            matched = v
            break

    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"No registered external model found for product={product} name={name}",
        )

    if not matched.file_path:
        raise HTTPException(
            status_code=404,
            detail=f"External model registered but file_path not set (req_no={matched.req_no})",
        )

    # 更新 last_used_at
    matched.last_used_at = datetime.utcnow()
    db.commit()
    db.refresh(matched)

    return matched


@router.get("/", response_model=List[ExternalModelOut])
async def list_external_models(
    product: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUser,
):
    """列出所有已登記的外部 model（status=registered）."""
    q = db.query(ModelVersion).filter(
        ModelVersion.status == "registered",
        ModelVersion.external_source.isnot(None),
    )
    if product:
        q = q.filter(ModelVersion.product == product)
    return q.order_by(ModelVersion.created_at.desc()).all()
