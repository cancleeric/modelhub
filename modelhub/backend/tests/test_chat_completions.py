"""
test_chat_completions.py — /v1/chat/completions 單元測試

測試三個主要 case：
1. 正常回應：slug 在 registry、有圖片、inference 成功
2. 無圖片 → 400
3. slug 不存在 → 404
"""

import base64
import json
import sys
import types
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unittest.mock import patch as _patch

# ---------------------------------------------------------------------------
# Module-level mock setup（先於任何 app import）
# ---------------------------------------------------------------------------

# mock parsers
if "parsers" not in sys.modules:
    parsers_mock = types.ModuleType("parsers")
    parsers_mock.parse_training_log = lambda arch, log_text: {"metrics": {}, "per_class": {}}
    sys.modules["parsers"] = parsers_mock

# mock models
if "models" not in sys.modules:
    models_mock = MagicMock()
    models_mock.SessionLocal = MagicMock()
    models_mock.Submission = MagicMock()
    models_mock.ModelVersion = MagicMock()
    models_mock.ApiKey = MagicMock()
    sys.modules["models"] = models_mock

# mock notifications
if "notifications" not in sys.modules:
    notif_mock = types.ModuleType("notifications")
    notif_mock.notify = AsyncMock(return_value=True)
    notif_mock.notify_event = AsyncMock(return_value=None)
    notif_mock.CTO_TARGET = "cto@hurricanecore.internal"
    sys.modules["notifications"] = notif_mock

import os
os.environ.setdefault("MODELHUB_API_KEY", "test-api-key")
os.environ.setdefault("SKIP_ROLE_CHECK", "true")

# ---------------------------------------------------------------------------
# Build minimal FastAPI test app
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.testclient import TestClient

# import router after mocks are in place
from routers.chat_completions import router as chat_router

app = FastAPI()
app.include_router(chat_router, prefix="/v1")

client = TestClient(app)

# 測試用 API Key header
HEADERS = {"X-Api-Key": "test-api-key"}


# ---------------------------------------------------------------------------
# autouse fixture：用 dependency_overrides 繞開 auth，避免跨 module import 汙染
# ---------------------------------------------------------------------------

async def _mock_api_key():
    """永遠通過驗證的假 API Key dependency。"""
    return "test-api-key"


@pytest.fixture(autouse=True)
def override_auth():
    """
    用 FastAPI dependency_overrides 替換 get_api_key，完全繞開 auth 模組。
    避免 test_integration_auth.py 在 collect 時重新 import auth 後汙染這批測試。
    """
    from routers.chat_completions import get_api_key
    app.dependency_overrides[get_api_key] = _mock_api_key
    yield
    app.dependency_overrides.clear()

# 最小 1x1 JPEG（白色像素），base64 encoded
_TINY_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
    b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00"
    b"\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00"
    b"\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81"
    b"\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19"
    b"\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86"
    b"\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4"
    b"\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2"
    b"\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9"
    b"\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5"
    b"\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd6"
    b"\xff\xd9"
)
_TINY_JPEG_B64 = base64.b64encode(_TINY_JPEG_BYTES).decode()
_DATA_URI = f"data:image/jpeg;base64,{_TINY_JPEG_B64}"

# inference server mock 回傳的 classification 結果
_MOCK_INFERENCE_RESULT = {
    "req_no": "MH-2026-010",
    "model": "mobilenetv2",
    "model_type": "classification",
    "prediction": "valve",
    "confidence": 0.9412,
    "all_scores": {"valve": 0.9412, "pump": 0.0588},
    "accuracy": 0.96,
}

# inference server /models 回傳的清單
_MOCK_MODELS_RESULT = {
    "models": [
        {"req_no": "MH-2026-010", "name": "PID 符號分類", "arch": "mobilenetv2", "source": "static"},
        {"req_no": "MH-2026-011", "name": "品質路由", "arch": "efficientnet_b0", "source": "static"},
    ]
}


# ---------------------------------------------------------------------------
# Helper：建立 httpx mock response
# ---------------------------------------------------------------------------

def _make_httpx_response(status_code: int, body: dict):
    """建立 httpx.Response mock 物件。"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatCompletionsNormal:
    """Case 1：正常推論回應。"""

    def test_normal_classification_response(self):
        """
        POST /v1/chat/completions，model=modelhub/mh-2026-010，帶 base64 圖片，
        應回傳 OpenAI chat.completion format，content 包含 class 和 confidence。
        """
        request_body = {
            "model": "modelhub/mh-2026-010",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "請辨識這張圖面的 PID 符號"},
                        {"type": "image_url", "image_url": {"url": _DATA_URI}},
                    ],
                }
            ],
        }

        models_response = _make_httpx_response(200, _MOCK_MODELS_RESULT)
        inference_response = _make_httpx_response(200, _MOCK_INFERENCE_RESULT)

        # AsyncClient.get → /models，AsyncClient.post → /predict/{slug}
        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(return_value=models_response)
        async_client_mock.post = AsyncMock(return_value=inference_response)

        with patch("routers.chat_completions.httpx.AsyncClient", return_value=async_client_mock):
            resp = client.post("/v1/chat/completions", json=request_body, headers=HEADERS)

        assert resp.status_code == 200
        data = resp.json()

        # OpenAI format 結構
        assert data["object"] == "chat.completion"
        assert data["model"] == "modelhub/mh-2026-010"
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["role"] == "assistant"

        content = choice["message"]["content"]
        assert "valve" in content
        assert "confidence" in content or "0.9412" in content

        # usage 欄位存在
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 0

        # id 格式
        assert data["id"].startswith("modelhub-")

    def test_slug_uppercase_normalized(self):
        """model slug 大寫應被 normalize 成小寫後呼叫 inference server。"""
        request_body = {
            "model": "modelhub/MH-2026-010",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _DATA_URI}},
                    ],
                }
            ],
        }

        models_response = _make_httpx_response(200, _MOCK_MODELS_RESULT)
        inference_response = _make_httpx_response(200, _MOCK_INFERENCE_RESULT)

        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(return_value=models_response)
        async_client_mock.post = AsyncMock(return_value=inference_response)

        with patch("routers.chat_completions.httpx.AsyncClient", return_value=async_client_mock):
            resp = client.post("/v1/chat/completions", json=request_body, headers=HEADERS)

        assert resp.status_code == 200
        # 確認 post 被呼叫時 URL 含 小寫 slug
        call_args = async_client_mock.post.call_args
        assert "mh-2026-010" in call_args[0][0]


class TestChatCompletionsNoImage:
    """Case 2：沒有圖片 → 400。"""

    def test_no_image_returns_400(self):
        """
        messages 中只有 text，沒有 image_url，應回傳 400。
        """
        request_body = {
            "model": "modelhub/mh-2026-010",
            "messages": [
                {
                    "role": "user",
                    "content": "請辨識這張圖面的 PID 符號",  # 純字串，無圖片
                }
            ],
        }

        models_response = _make_httpx_response(200, _MOCK_MODELS_RESULT)

        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(return_value=models_response)

        with patch("routers.chat_completions.httpx.AsyncClient", return_value=async_client_mock):
            resp = client.post("/v1/chat/completions", json=request_body, headers=HEADERS)

        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()

    def test_no_image_in_content_parts_returns_400(self):
        """
        content 是 parts list 但都是 text type，應回傳 400。
        """
        request_body = {
            "model": "modelhub/mh-2026-010",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "請辨識"},
                        {"type": "text", "text": "這張圖面"},
                    ],
                }
            ],
        }

        models_response = _make_httpx_response(200, _MOCK_MODELS_RESULT)

        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(return_value=models_response)

        with patch("routers.chat_completions.httpx.AsyncClient", return_value=async_client_mock):
            resp = client.post("/v1/chat/completions", json=request_body, headers=HEADERS)

        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()


class TestChatCompletionsSlugNotFound:
    """Case 3：slug 不存在 → 404。"""

    def test_unknown_slug_returns_404(self):
        """
        model="modelhub/mh-9999-999"，在 inference server registry 找不到，應回 404。
        """
        request_body = {
            "model": "modelhub/mh-9999-999",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _DATA_URI}},
                    ],
                }
            ],
        }

        # /models 回傳只有 010 和 011
        models_response = _make_httpx_response(200, _MOCK_MODELS_RESULT)

        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(return_value=models_response)

        with patch("routers.chat_completions.httpx.AsyncClient", return_value=async_client_mock):
            resp = client.post("/v1/chat/completions", json=request_body, headers=HEADERS)

        assert resp.status_code == 404
        assert "mh-9999-999" in resp.json()["detail"]

    def test_inference_server_unavailable_returns_503(self):
        """
        inference server 無法連線，應回 503。
        """
        import httpx as _httpx

        request_body = {
            "model": "modelhub/mh-2026-010",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _DATA_URI}},
                    ],
                }
            ],
        }

        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(side_effect=_httpx.ConnectError("Connection refused"))

        with patch("routers.chat_completions.httpx.AsyncClient", return_value=async_client_mock):
            resp = client.post("/v1/chat/completions", json=request_body, headers=HEADERS)

        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
