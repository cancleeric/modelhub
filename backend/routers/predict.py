"""
predict.py — 推論 proxy router

路由：
  GET  /api/predict/available          列出可用推論模型
  POST /api/predict/{req_no}           上傳圖片取得預測結果

實作：
  本 router 作為 proxy，將請求轉發到 host 推論 server（port 8951）。
  推論 server 跑在 host Python（torch 環境），避免在容器內安裝 torch。

推論 server 啟動方式：
  python3 ~/HurricaneCore/modelhub/inference-server/inference_server.py

認證：同現有 API，使用 CurrentUserOrApiKey。
"""

import os
import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from auth import CurrentUserOrApiKey

router = APIRouter()

# 推論 server URL（容器內透過 host.docker.internal 呼叫 host port 8951）
# Mac Docker Desktop 支援 host.docker.internal
_INFER_HOST = os.getenv("MODELHUB_INFER_URL", "http://host.docker.internal:8951")

# M-R2-2: 上傳安全限制
_MAX_UPLOAD_BYTES = int(os.getenv("MODELHUB_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10MB
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


async def _call_infer(method: str, path: str, **kwargs) -> httpx.Response:
    """呼叫推論 server，統一處理連線錯誤。"""
    url = f"{_INFER_HOST}{path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.request(method, url, **kwargs)
            return resp
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Inference server unavailable. Please start modelhub inference-server on port 8951.",
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Inference server timeout")


@router.get("/available")
async def list_available_models(
    _user: dict = CurrentUserOrApiKey,
):
    """列出所有可用推論模型（verdict=pass）。"""
    resp = await _call_infer("GET", "/models")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Inference server error")
    return JSONResponse(content=resp.json())


@router.post("/{req_no}")
async def predict(
    req_no: str,
    file: UploadFile = File(...),
    _user: dict = CurrentUserOrApiKey,
):
    """
    上傳圖片，回傳模型預測結果。

    - req_no: 工單編號（MH-2026-010 / MH-2026-011 等）
    - file: multipart/form-data，field name `file`，一張圖片
    """
    # M-R2-2: content-type 白名單（避免任意檔案上傳）
    ctype = (file.content_type or "").lower()
    if ctype not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content-type '{file.content_type}'. Allowed: {sorted(_ALLOWED_CONTENT_TYPES)}",
        )

    contents = await file.read()

    # M-R2-2: 大小限制（防 DoS）
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)} bytes, max {_MAX_UPLOAD_BYTES})",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # filename sanitize：移除路徑分隔符，僅保留 basename
    safe_filename = os.path.basename(file.filename or "image.jpg").replace("\0", "")
    if not safe_filename:
        safe_filename = "image.jpg"

    resp = await _call_infer(
        "POST",
        f"/predict/{req_no}",
        files={"file": (safe_filename, contents, ctype or "image/jpeg")},
    )

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Model not available for prediction")
    if resp.status_code == 400:
        raise HTTPException(status_code=400, detail=resp.json().get("detail", "Bad request"))
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Inference server error")

    return JSONResponse(content=resp.json())
