"""
chat_completions.py — OpenAI-compatible /v1/chat/completions endpoint

讓 brain-llm-connector 的 LlmHandler（AsyncOpenAI client）可以直接呼叫 ModelHub，
零改動即可取得圖像分類推論結果，或呼叫文字生成模型。

支援的 model 格式：
  "modelhub/mh-2026-010"  → strip 前綴得 slug "mh-2026-010"
  "modelhub/MH-2026-010"  → slug 統一 lower-case

流程（有圖片 → 圖像路徑；無圖片 → 文字路徑）：
  有圖片：
    1. 解析 model slug
    2. 確認 slug 在 inference server registry 裡
    3. 從 messages 找最後一個 image_url content part
    4. POST multipart/form-data → {MODELHUB_INFER_URL}/predict/{slug}
    5. 把 classification 結果包裝成 OpenAI chat.completion response
  無圖片（純文字）：
    1. 解析 model slug
    2. POST JSON → {MODELHUB_INFER_URL}/generate/{slug}（目前回傳 501）
    3. 未來文字生成模型上線後，此路徑自動生效

錯誤：
  404 — slug 不在 inference server 支援清單
  501 — 純文字請求，文字生成尚未支援
  503 — inference server 無回應
"""

import base64
import io
import logging
import os
import time
import uuid
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import get_api_key

logger = logging.getLogger("modelhub.chat_completions")

# host inference server（跑在 host Python，不在容器內）
# 容器內需設 MODELHUB_INFER_URL=http://host.docker.internal:8951 才能連到 host
INFERENCE_SERVER_URL = os.getenv("MODELHUB_INFER_URL", "http://localhost:8951")
INFERENCE_TIMEOUT = 30.0

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas（OpenAI-compatible subset）
# ---------------------------------------------------------------------------

class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str                              # "text" | "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str
    # content 可以是純字串或 content parts list
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_slug(model: str) -> str:
    """
    從 model 欄位解析 slug。
    "modelhub/mh-2026-010" → "mh-2026-010"
    "modelhub/MH-2026-010" → "mh-2026-010"
    """
    if model.startswith("modelhub/"):
        return model[len("modelhub/"):].lower()
    return model.lower()


def _has_image(messages: list[Message]) -> bool:
    """
    判斷 messages 中是否包含圖片（image_url content part 或 data URI）。
    """
    import re as _re

    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            if _re.search(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', content):
                return True
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url" and part.get("image_url", {}).get("url"):
                        return True
                elif hasattr(part, "type"):
                    if part.type == "image_url" and part.image_url:
                        return True
    return False


def _extract_image(messages: list[Message]) -> bytes:
    """
    從 messages 找最後一個 image_url content part，回傳圖片 bytes。
    支援：
      - data:image/...;base64,<data>  (base64 data URI)
      - http/https URL
    沒找到圖片 → raise HTTPException(400)
    """
    image_bytes: Optional[bytes] = None

    import re as _re

    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            # 支援純文字中嵌入的 data URI（例如 brain-cloud 轉發的 new_message content）
            match = _re.search(r'(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)', content)
            if match:
                image_bytes = _fetch_image_bytes(match.group(1))
            continue
        if isinstance(content, list):
            for part in content:
                # part 可能是 dict（pydantic 未解析）或 ContentPart
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "image_url":
                        img_url_obj = part.get("image_url", {})
                        url = img_url_obj.get("url", "") if isinstance(img_url_obj, dict) else ""
                        if url:
                            image_bytes = _fetch_image_bytes(url)
                elif hasattr(part, "type"):
                    if part.type == "image_url" and part.image_url:
                        image_bytes = _fetch_image_bytes(part.image_url.url)

    if image_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="No image found in messages. Please include an image_url content part.",
        )
    return image_bytes


def _fetch_image_bytes(url: str) -> bytes:
    """
    從 data URI 或 http/https URL 取得圖片 bytes。
    """
    if url.startswith("data:"):
        # data:image/jpeg;base64,<data>
        try:
            if "," not in url:
                raise ValueError("Invalid data URI")
            b64_data = url.split(",", 1)[1]
            return base64.b64decode(b64_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data URI: {e}")
    elif url.startswith("http://") or url.startswith("https://"):
        try:
            resp = httpx.get(url, timeout=10.0, follow_redirects=True)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to fetch image URL (HTTP {resp.status_code})",
                )
            return resp.content
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot fetch image URL: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported image URL scheme: {url[:30]}")


async def _check_slug_in_registry(slug: str) -> None:
    """
    確認 slug 在 inference server 的 /models 清單中。
    無法連線 → 503；slug 找不到 → 404。
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{INFERENCE_SERVER_URL}/models")
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Inference server is unavailable (connection refused on port 8951)",
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Inference server error: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail=f"Inference server /models returned HTTP {resp.status_code}",
        )

    data = resp.json()
    models = data.get("models", [])
    req_nos = {(m.get("req_no") or "").lower() for m in models}

    if slug not in req_nos:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{slug}' not found in inference server registry",
        )


async def _call_inference(slug: str, img_bytes: bytes) -> dict:
    """
    POST 圖片到 inference server /predict/{slug}，回傳 response JSON。
    503 on connection error，timeout 30s。
    """
    try:
        async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
            resp = await client.post(
                f"{INFERENCE_SERVER_URL}/predict/{slug}",
                files={"file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Inference server is unavailable (connection refused on port 8951)",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=503,
            detail="Inference server timed out (> 30s)",
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Inference server error: {e}")

    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{slug}' not found in inference server",
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail=f"Inference server returned HTTP {resp.status_code}: {resp.text[:200]}",
        )

    return resp.json()


def _format_response(model: str, result: dict) -> dict:
    """
    把 inference server classification 結果包裝成 OpenAI chat.completion response。
    """
    prediction = result.get("prediction", "unknown")
    confidence = result.get("confidence", 0.0)
    model_type = result.get("model_type", "classification")

    if model_type == "detection":
        detections = result.get("detections", [])
        count = result.get("count", len(detections))
        content = f"偵測結果：找到 {count} 個物件"
        if detections:
            items = [
                f"{d['label']}（confidence={d['confidence']:.4f}）"
                for d in detections[:5]  # 最多顯示 5 個
            ]
            content += "：" + "、".join(items)
    else:
        content = f"辨識結果：class={prediction}, confidence={confidence:.4f}"

    # 計算 completion_tokens（簡單估算：每個字元約 1 token）
    completion_tokens = max(1, len(content) // 3)

    return {
        "id": f"modelhub-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    _api_key: str = Depends(get_api_key),
):
    """
    OpenAI-compatible chat completions endpoint。

    brain-llm-connector 的 LlmHandler（AsyncOpenAI client）可直接呼叫，零改動。

    - model: "modelhub/<slug>"（例如 "modelhub/mh-2026-010"）
    - messages 含圖片 → 走圖像分類路徑（/predict/{slug}）
    - messages 純文字 → 走文字生成路徑（/generate/{slug}，目前回傳 501）
    - 認證：X-Api-Key header

    回傳 OpenAI chat.completion format。
    """
    slug = _parse_slug(req.model)
    logger.info("chat_completions: model=%s slug=%s", req.model, slug)

    if _has_image(req.messages):
        # ── 圖像路徑 ──────────────────────────────────────────────────────
        # 1. 確認 slug 存在於 inference server registry
        await _check_slug_in_registry(slug)

        # 2. 從 messages 提取圖片
        img_bytes = _extract_image(req.messages)

        # 3. 呼叫 inference server
        result = await _call_inference(slug, img_bytes)
        logger.info("chat_completions: slug=%s prediction=%s confidence=%s",
                    slug, result.get("prediction"), result.get("confidence"))

        # 4. 包裝 OpenAI response
        return _format_response(req.model, result)

    else:
        # ── 文字生成路徑 ──────────────────────────────────────────────────
        # 純文字請求，嘗試呼叫 /generate/{slug}。
        # 目前 inference server 尚未實作此 endpoint，回傳 501。
        # 當文字生成模型上線後，此路徑會自動通。
        logger.info("chat_completions: 純文字請求 → /generate/%s", slug)
        try:
            async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
                resp = await client.post(
                    f"{INFERENCE_SERVER_URL}/generate/{slug}",
                    json={"messages": [m.model_dump() for m in req.messages]},
                )
            if resp.status_code == 200:
                result = resp.json()
                return _format_response(req.model, result)
            # inference server 回 404 或 501 → 統一回 501
        except httpx.ConnectError:
            pass  # inference server 未啟動，也走 501
        except Exception as e:
            logger.warning("chat_completions: /generate/%s 呼叫異常: %s", slug, e)

        raise HTTPException(
            status_code=501,
            detail="text generation not yet supported for this model",
        )
