"""
M12 N4 — notifications.py webhook dispatch integration test

驗收：mock brain-cloud endpoint 收到正確 payload

執行：
  cd modelhub/backend
  python3 -m pytest tests/test_webhook_notification.py -v --tb=short
"""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("BRAIN_CLOUD_URL", "http://brain-cloud-dev:8932")
os.environ.setdefault("BRAIN_API_KEY", "test-internal-key-32plus-chars-aaaa")
os.environ.setdefault("MODELHUB_CTO_USERNAME", "cto@hurricanecore.internal")


def _make_submission(req_no: str, submitter: str = "user@example.com", map50: float = 0.75):
    """建立 mock submission 物件。"""
    sub = MagicMock()
    sub.req_no = req_no
    sub.submitter = submitter
    sub.req_name = "Test Model"
    sub.product = "test-product"
    sub.resubmit_count = 0
    sub.latest_model_version = "v1.0.0"
    return sub


class TestWebhookDispatchIntegration:
    """N4: mock brain-cloud endpoint，驗 notify_event 送出正確 payload。"""

    @pytest.mark.asyncio
    async def test_training_complete_dispatches_webhook(self):
        """training_complete 觸發 brain-cloud dispatch-webhook 呼叫。"""
        from notifications import _dispatch_brain_webhook

        captured = {}

        async def mock_post(url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            resp = MagicMock()
            resp.status_code = 200
            resp.json = lambda: {
                "event_type": "training_complete",
                "subscriptions_found": 1,
                "success_count": 1,
                "fail_count": 0,
                "dispatched_at": 1714000000.0,
            }
            return resp

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=mock_post)

        sub = _make_submission("MH-2026-019")

        with patch("notifications.httpx.AsyncClient", return_value=mock_client):
            await _dispatch_brain_webhook(
                event_type="training_complete",
                submission=sub,
                model_version="v1.0.0",
                map50=0.7455,
            )

        assert "url" in captured, "brain-cloud POST 應被呼叫"
        assert "/internal/dispatch-webhook" in captured["url"]
        assert captured["json"]["event_type"] == "training_complete"
        assert captured["json"]["req_no"] == "MH-2026-019"
        assert captured["json"]["model_version"] == "v1.0.0"
        assert abs(captured["json"]["map50"] - 0.7455) < 0.0001
        assert captured["headers"]["X-Internal-Key"] == os.environ["BRAIN_API_KEY"]

    @pytest.mark.asyncio
    async def test_training_failed_dispatches_webhook(self):
        """training_failed 觸發 brain-cloud dispatch-webhook 呼叫。"""
        from notifications import _dispatch_brain_webhook

        captured = {}

        async def mock_post(url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            resp = MagicMock()
            resp.status_code = 200
            resp.json = lambda: {
                "event_type": "training_failed",
                "subscriptions_found": 0,
                "success_count": 0,
                "fail_count": 0,
                "dispatched_at": 1714000000.0,
            }
            return resp

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=mock_post)

        sub = _make_submission("MH-2026-020")

        with patch("notifications.httpx.AsyncClient", return_value=mock_client):
            await _dispatch_brain_webhook(
                event_type="training_failed",
                submission=sub,
            )

        assert captured["json"]["event_type"] == "training_failed"
        assert captured["json"]["req_no"] == "MH-2026-020"

    @pytest.mark.asyncio
    async def test_dispatch_silent_when_no_brain_api_key(self):
        """BRAIN_API_KEY 未設時應靜默跳過，不 raise。"""
        original_key = os.environ.get("BRAIN_API_KEY", "")
        os.environ["BRAIN_API_KEY"] = ""

        try:
            # 即使 brain-cloud 不可達也不應 raise
            from notifications import _dispatch_brain_webhook
            import importlib
            import notifications
            importlib.reload(notifications)  # 重新載入讓 BRAIN_API_KEY 生效

            sub = _make_submission("MH-TEST")
            # 應靜默完成
            await notifications._dispatch_brain_webhook(
                event_type="training_complete",
                submission=sub,
            )
        finally:
            os.environ["BRAIN_API_KEY"] = original_key

    @pytest.mark.asyncio
    async def test_notify_event_training_complete_includes_webhook(self):
        """notify_event('training_complete') 流程包含 CMC 通知 + webhook dispatch。"""
        from notifications import notify_event

        cmc_calls = []
        webhook_calls = []

        async def mock_notify(to, message):
            cmc_calls.append({"to": to, "message": message})
            return True

        async def mock_dispatch(event_type, submission, model_version=None, map50=None):
            webhook_calls.append({
                "event_type": event_type,
                "req_no": submission.req_no,
                "model_version": model_version,
                "map50": map50,
            })

        sub = _make_submission("MH-2026-021", submitter="user@example.com")

        with patch("notifications.notify", new=mock_notify), \
             patch("notifications._dispatch_brain_webhook", new=mock_dispatch):
            await notify_event("training_complete", sub)

        # CMC 通知應有 CTO + submitter
        assert len(cmc_calls) >= 1
        cmc_targets = {c["to"] for c in cmc_calls}
        assert "cto@hurricanecore.internal" in cmc_targets

        # webhook dispatch 應被呼叫
        assert len(webhook_calls) == 1
        assert webhook_calls[0]["event_type"] == "training_complete"
        assert webhook_calls[0]["req_no"] == "MH-2026-021"
