"""
routers/attachments.py — M22 附件上傳 / 下載

Endpoints:
  POST   /api/submissions/{req_no}/attachments   multipart upload
  GET    /api/attachments/{id}                   streaming download
  DELETE /api/attachments/{id}                   物理刪除（只 uploader / superadmin）

儲存路徑：~/HurricaneCore/docker-data/modelhub/attachments/<req_no>/<uuid>.<ext>
大小限制：10 MB / file
MIME 白名單：image/*, text/*, application/pdf, application/json, application/zip
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth import CurrentUserOrApiKey, _ROLE_CLAIM_KEY, _SKIP_ROLE_CHECK
from models import Submission, SubmissionAttachment, get_db

_logger = logging.getLogger("modelhub.routers.attachments")

# 大小限制：10 MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# MIME 白名單（前綴比對）
_ALLOWED_MIME_PREFIXES = ("image/", "text/")
_ALLOWED_MIME_EXACT = {
    "application/pdf",
    "application/json",
    "application/zip",
}

# 附件儲存根目錄（bind mount 到 host）
_ATTACHMENT_BASE_DIR = Path(
    os.getenv(
        "ATTACHMENT_BASE_DIR",
        os.path.expanduser("~/HurricaneCore/docker-data/modelhub/attachments"),
    )
)

# 特權 role
_PRIVILEGED_ROLES = {"reviewer", "cto", "superadmin"}

router = APIRouter()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_user_role(current_user: dict) -> str:
    if not current_user:
        return "submitter"
    sub = str(current_user.get("sub", ""))
    if sub.startswith("api_key:"):
        return "reviewer"
    return current_user.get(_ROLE_CLAIM_KEY) or "submitter"


def _get_user_email(current_user: dict) -> str:
    if not current_user:
        return "unknown"
    return (
        current_user.get("email")
        or current_user.get("preferred_username")
        or current_user.get("sub")
        or "unknown"
    )


def _is_privileged(current_user: dict) -> bool:
    return _SKIP_ROLE_CHECK or _get_user_role(current_user) in _PRIVILEGED_ROLES


def _validate_mime(mime: str) -> bool:
    for prefix in _ALLOWED_MIME_PREFIXES:
        if mime.startswith(prefix):
            return True
    return mime in _ALLOWED_MIME_EXACT


def _assert_submission_access(req_no: str, db: Session, current_user: dict) -> Submission:
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail=f"Submission {req_no} not found")
    if _is_privileged(current_user):
        return obj
    user_company = current_user.get("company") or current_user.get("tenant")
    if user_company and obj.company and user_company.lower() != obj.company.lower():
        raise HTTPException(status_code=403, detail="Tenant mismatch: access denied")
    return obj


# ---------------------------------------------------------------------------
# POST /api/submissions/{req_no}/attachments
# ---------------------------------------------------------------------------

@router.post("/submissions/{req_no}/attachments", status_code=201)
async def upload_attachment(
    req_no: str,
    file: UploadFile = File(...),
    comment_id: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    上傳附件到工單（multipart/form-data）。
    - 單檔 10MB 限制
    - MIME 白名單驗證
    - 儲存到 bind mount 目錄
    """
    _assert_submission_access(req_no, db, current_user)

    # 讀取並驗證大小
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File too large: {len(content)} bytes (max {MAX_FILE_SIZE_BYTES} bytes / 10 MB)",
        )

    # 驗證 MIME
    content_type = file.content_type or "application/octet-stream"
    if not _validate_mime(content_type):
        raise HTTPException(
            status_code=422,
            detail=(
                f"MIME type '{content_type}' not allowed. "
                "Allowed: image/*, text/*, application/pdf, application/json, application/zip"
            ),
        )

    # 建立儲存目錄
    req_dir = _ATTACHMENT_BASE_DIR / req_no
    req_dir.mkdir(parents=True, exist_ok=True)

    # 組合唯一檔名（保留副檔名）
    original_name = file.filename or "upload"
    ext = Path(original_name).suffix
    unique_name = f"{uuid.uuid4()}{ext}"
    storage_path = str(req_dir / unique_name)

    # 寫檔
    with open(storage_path, "wb") as f:
        f.write(content)

    uploader = _get_user_email(current_user)
    now = datetime.utcnow()

    attachment = SubmissionAttachment(
        req_no=req_no,
        comment_id=comment_id,
        filename=original_name,
        size_bytes=len(content),
        mime_type=content_type,
        storage_path=storage_path,
        uploaded_by=uploader,
        uploaded_at=now,
    )
    db.add(attachment)
    db.commit()
    db.refresh(attachment)

    _logger.info(
        "attachment uploaded: id=%d req_no=%s file=%s size=%d mime=%s",
        attachment.id, req_no, original_name, len(content), content_type,
    )

    return {
        "id": attachment.id,
        "filename": attachment.filename,
        "size_bytes": attachment.size_bytes,
        "mime_type": attachment.mime_type,
        "uploaded_by": attachment.uploaded_by,
        "uploaded_at": attachment.uploaded_at.isoformat(),
        "url": f"/api/attachments/{attachment.id}",
    }


# ---------------------------------------------------------------------------
# GET /api/attachments/{id}
# ---------------------------------------------------------------------------

@router.get("/attachments/{attachment_id}")
async def download_attachment(
    attachment_id: int,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    串流下載附件。
    tenant 隔離：只有同 tenant 的使用者（或 reviewer+）可下載。
    """
    att = db.query(SubmissionAttachment).filter(SubmissionAttachment.id == attachment_id).first()
    if not att:
        raise HTTPException(status_code=404, detail="Attachment not found")

    # tenant 隔離：查對應 submission 的 company
    _assert_submission_access(att.req_no, db, current_user)

    if not os.path.exists(att.storage_path):
        raise HTTPException(status_code=404, detail="Attachment file not found on disk")

    return FileResponse(
        path=att.storage_path,
        filename=att.filename,
        media_type=att.mime_type,
        headers={"Content-Disposition": f'attachment; filename="{att.filename}"'},
    )


# ---------------------------------------------------------------------------
# DELETE /api/attachments/{id}
# ---------------------------------------------------------------------------

@router.delete("/attachments/{attachment_id}", status_code=204)
async def delete_attachment(
    attachment_id: int,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    物理刪除附件（只有 uploader 或 superadmin）。
    """
    att = db.query(SubmissionAttachment).filter(SubmissionAttachment.id == attachment_id).first()
    if not att:
        raise HTTPException(status_code=404, detail="Attachment not found")

    uploader_email = _get_user_email(current_user)
    user_role = _get_user_role(current_user)

    if uploader_email != att.uploaded_by and user_role != "superadmin" and not _SKIP_ROLE_CHECK:
        raise HTTPException(status_code=403, detail="Only the uploader or superadmin can delete this attachment")

    # 物理刪除檔案
    if os.path.exists(att.storage_path):
        os.remove(att.storage_path)
        _logger.info("attachment file deleted: %s", att.storage_path)

    db.delete(att)
    db.commit()
    return None
