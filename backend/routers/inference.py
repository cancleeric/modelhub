"""
inference.py — 推論 API router（供跨公司機器對機器呼叫）

路由：
  GET  /v1/health              健康檢查（不需 auth）
  GET  /v1/models              列出所有已驗收模型（X-API-Key）
  POST /v1/detect              YOLO 物件偵測推論（X-API-Key）

認證：X-API-Key header（同現有 auth.py）。/v1/health 不需認證。

工單：#202604180008（AICAD 推論 API 需求）
"""

import base64
import io
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_api_key
from models import ModelVersion, Submission, get_db

logger = logging.getLogger("modelhub.inference")

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    model_name: str                       # req_no（如 "MH-2026-010"）或 model version id（int 字串）
    image_url: Optional[str] = None
    image_b64: Optional[str] = None       # base64 encoded image
    conf_threshold: float = 0.25


class DetectResponse(BaseModel):
    boxes: list[dict]                     # [{x1,y1,x2,y2,conf,class_id,class_name}]
    model_version: str
    model_name: str
    latency_ms: float
    image_size: list[int]                 # [w, h]


class ModelInfo(BaseModel):
    id: int
    req_no: str
    arch: Optional[str]
    map50: Optional[float]
    class_names: Optional[list[str]]
    version_tag: str
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_bytes(req: DetectRequest) -> bytes:
    """從 image_b64 或 image_url 取得圖片 bytes。"""
    if req.image_b64:
        try:
            # 支援 data:image/...;base64,<data> 格式
            b64_data = req.image_b64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            return base64.b64decode(b64_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    elif req.image_url:
        try:
            import httpx
            resp = httpx.get(req.image_url, timeout=10.0, follow_redirects=True)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to fetch image_url (HTTP {resp.status_code})",
                )
            return resp.content
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot fetch image_url: {e}")
    else:
        raise HTTPException(status_code=400, detail="Must provide image_b64 or image_url")


def _resolve_model_version(model_name: str, db: Session) -> ModelVersion:
    """
    根據 model_name 查 ModelVersion。
    支援：req_no（如 "MH-2026-010"）或 id（整數字串）。
    """
    # 嘗試用 id 查
    if model_name.isdigit():
        mv = db.query(ModelVersion).filter(ModelVersion.id == int(model_name)).first()
        if mv:
            return mv
        raise HTTPException(status_code=404, detail=f"ModelVersion id={model_name} not found")

    # 用 req_no 查（大小寫不敏感，取 is_current=True 優先，否則取最新 pass）
    q = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.req_no == model_name.upper(),
            ModelVersion.pass_fail == "pass",
        )
        .order_by(ModelVersion.is_current.desc(), ModelVersion.created_at.desc())
    )
    mv = q.first()
    if mv:
        return mv

    # 再試 req_no 不分大小寫（SQLite LIKE）
    q2 = (
        db.query(ModelVersion)
        .filter(ModelVersion.req_no.ilike(model_name))
        .order_by(ModelVersion.is_current.desc(), ModelVersion.created_at.desc())
    )
    mv = q2.first()
    if mv:
        return mv

    raise HTTPException(
        status_code=404,
        detail=f"No accepted ModelVersion found for '{model_name}'",
    )


def _get_class_names(mv: ModelVersion, db: Session) -> list[str]:
    """
    從 Submission.class_list 取 class 名稱（JSON array 或逗號分隔）。
    找不到回傳空 list。
    """
    if not mv.req_no:
        return []
    sub = db.query(Submission).filter(Submission.req_no == mv.req_no).first()
    if not sub or not sub.class_list:
        return []
    raw = sub.class_list.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    # 逗號分隔
    return [c.strip() for c in raw.split(",") if c.strip()]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def inference_health():
    """健康檢查，不需認證。"""
    return {"status": "ok", "service": "modelhub-inference"}


@router.get("/models")
def list_accepted_models(
    db: Session = Depends(get_db),
    _api_key: str = Depends(get_api_key),
):
    """
    列出所有已驗收（pass_fail=pass）的 ModelVersion。
    回傳欄位：id, req_no, arch, map50, class_names, version_tag, created_at。
    """
    rows = (
        db.query(ModelVersion)
        .filter(ModelVersion.pass_fail == "pass")
        .order_by(ModelVersion.created_at.desc())
        .all()
    )
    result = []
    for mv in rows:
        class_names = _get_class_names(mv, db)
        result.append({
            "id": mv.id,
            "req_no": mv.req_no,
            "arch": mv.arch,
            "map50": mv.map50_actual if mv.map50_actual is not None else mv.map50,
            "class_names": class_names,
            "version_tag": mv.version,
            "created_at": mv.created_at.isoformat() if mv.created_at else None,
        })
    return {"models": result, "count": len(result)}


@router.post("/detect", response_model=DetectResponse)
def detect(
    req: DetectRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(get_api_key),
):
    """
    YOLO 物件偵測推論。

    - model_name: req_no（如 "MH-2026-010"）或 ModelVersion.id 字串
    - image_b64: base64 編碼圖片（優先）
    - image_url: 圖片 URL（備用）
    - conf_threshold: 信心度閾值（預設 0.25）

    回傳 bbox 格式：x1y1x2y2 絕對像素座標。
    timeout: 5 秒（inference 超時回 504）。
    """
    # 1. 解析模型版本
    mv = _resolve_model_version(req.model_name, db)

    # 2. 確認模型檔案存在
    if not mv.file_path:
        raise HTTPException(
            status_code=404,
            detail="model file not found，請聯絡 ModelHub 管理員",
        )

    import os
    from pathlib import Path

    model_path = Path(mv.file_path)
    # 若是相對路徑，嘗試在 training 目錄下尋找
    if not model_path.is_absolute():
        training_base = Path(
            os.environ.get(
                "MODELHUB_TRAINING_BASE",
                "/Users/yinghaowang/HurricaneCore/modelhub/training",
            )
        )
        # 先在 training/<req_no_lower>/ 下找
        candidate = training_base / mv.req_no.lower() / model_path
        if candidate.exists():
            model_path = candidate
        else:
            # 直接在 training_base 下找
            candidate2 = training_base / model_path
            if candidate2.exists():
                model_path = candidate2

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="model file not found，請聯絡 ModelHub 管理員",
        )

    # 3. 讀取圖片
    img_bytes = _load_image_bytes(req)

    # 4. YOLO 推論（timeout 5s）
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="ultralytics 套件未安裝，無法執行 YOLO 推論",
        )

    from PIL import Image as PILImage
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("inference timeout")

    t0 = time.time()
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)

        model = YOLO(str(model_path))
        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = pil_img.size

        results = model(pil_img, conf=req.conf_threshold, verbose=False)
        signal.alarm(0)  # 取消 alarm
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout (> 5s)")
    except HTTPException:
        raise
    except Exception as e:
        signal.alarm(0)
        logger.error("YOLO inference error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    latency_ms = round((time.time() - t0) * 1000, 1)

    # 5. 整理結果
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = r.names.get(cls_id, str(cls_id))
            boxes.append({
                "x1": round(xyxy[0], 2),
                "y1": round(xyxy[1], 2),
                "x2": round(xyxy[2], 2),
                "y2": round(xyxy[3], 2),
                "conf": round(conf, 6),
                "class_id": cls_id,
                "class_name": class_name,
            })

    return DetectResponse(
        boxes=boxes,
        model_version=mv.version,
        model_name=mv.model_name,
        latency_ms=latency_ms,
        image_size=[img_w, img_h],
    )
