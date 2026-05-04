"""
LLM Handler — 呼叫 ModelHub /v1/chat/completions 的邏輯

modelhub-connector 橋接 brain-cloud 到 ModelHub backend。
ModelHub 認證：X-Api-Key header（env MODELHUB_API_KEY）。

注意：ModelHub /v1/chat/completions 目前不支援真正串流（回傳完整 response）。
串流模式：把完整 response 拆成單一 stream_token + stream_end 回傳給 brain-cloud。
非串流模式：直接回傳 llm_response。
"""

import json
import os
import time

import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger("modelhub_connector.handler")

MODELHUB_API_URL = os.environ.get("MODELHUB_API_URL", "http://localhost:8950")
MODELHUB_API_KEY = os.environ.get("MODELHUB_API_KEY", "")

REQUEST_TIMEOUT = int(os.environ.get("LLM_REQUEST_TIMEOUT", "60"))


def _make_client() -> AsyncOpenAI:
    """
    建立 AsyncOpenAI client，指向 ModelHub /v1。
    認證走 X-Api-Key header（ModelHub 的 get_api_key dependency 讀此 header）。
    api_key 欄位填任意非空字串以滿足 openai SDK 要求（不用於實際認證）。
    """
    return AsyncOpenAI(
        api_key="modelhub-connector",  # SDK 必填，不用於認證
        base_url=f"{MODELHUB_API_URL}/v1",
        timeout=REQUEST_TIMEOUT,
        default_headers={
            "x-api-key": MODELHUB_API_KEY,
        },
    )


class LlmHandler:
    """封裝呼叫 ModelHub /v1/chat/completions 的邏輯。"""

    async def handle_stream(
        self,
        ws,
        request_id: str,
        model: str,
        messages: list[dict],
        call_type: str = "chat",
        tenant_id: str = "",
        agent_id: str = "",
    ):
        """
        串流模式：ModelHub 不支援真正串流，把完整 response 包成
        單一 stream_token + stream_end 回傳給 brain-cloud。
        """
        client = _make_client()
        start_ts = time.monotonic()

        logger.info(
            "stream 呼叫 → ModelHub",
            request_id=request_id,
            model=model,
            tenant_id=tenant_id,
        )

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,  # ModelHub 不支援真正串流
        )

        usage = resp.usage
        content = ""
        if resp.choices:
            content = resp.choices[0].message.content or ""

        duration_ms = int((time.monotonic() - start_ts) * 1000)
        logger.info(
            "ModelHub 回應完成，送 stream_token + stream_end",
            request_id=request_id,
            model=model,
            content_len=len(content),
            duration_ms=duration_ms,
        )

        # 單一 token 包含完整內容
        try:
            await ws.send(json.dumps({
                "type": "stream_token",
                "request_id": request_id,
                "token": content,
            }))
        except Exception as send_err:
            logger.error(
                "stream_token 送出失敗（WS 已斷？）",
                request_id=request_id,
                error=str(send_err),
            )
            raise

        try:
            await ws.send(json.dumps({
                "type": "stream_end",
                "request_id": request_id,
                "usage": {
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                },
            }))
        except Exception as send_err:
            logger.error(
                "stream_end 送出失敗（WS 已斷？）",
                request_id=request_id,
                error=str(send_err),
            )
            raise

        logger.info(
            "stream 完成",
            request_id=request_id,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def handle_non_stream(
        self,
        ws,
        request_id: str,
        model: str,
        messages: list[dict],
        call_type: str = "chat",
        tenant_id: str = "",
        agent_id: str = "",
    ):
        """
        非串流模式：等待完整 response 後送 llm_response。
        """
        client = _make_client()

        logger.info(
            "non-stream 呼叫 → ModelHub",
            request_id=request_id,
            model=model,
            tenant_id=tenant_id,
        )

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
        )

        usage = resp.usage
        await ws.send(json.dumps({
            "type": "llm_response",
            "request_id": request_id,
            "response": resp.model_dump(),
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        }))

        logger.info(
            "non-stream 完成",
            request_id=request_id,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
