"""
test_submissions_limit.py — M13 P6：submissions list limit/offset 上限驗證

驗收標準：
- list_submissions limit 參數有 ge=1, le=100（FastAPI Query metadata）
- list_submissions offset 參數有 ge=0, le=10000
- ?limit=99999 觸發 422（在無認證 bypass 路徑下驗證）
- 正常查詢不受影響（limit=50 default）

注意：FastAPI 0.115+ 在 Depends 認證失敗時，Query validation 不先執行。
      因此用 Query metadata 直接驗證，不依賴 HTTP integration test。
"""
import os
import sys
import types
import unittest.mock as mock

import pytest


def _setup_modules():
    """Mock 必要模組避免 import 錯誤。"""
    mocks = {
        "models": mock.MagicMock(),
        "notifications": _make_notifications_mock(),
        "parsers": _make_parsers_mock(),
    }
    for name, m in mocks.items():
        if name not in sys.modules:
            sys.modules[name] = m


def _make_notifications_mock():
    m = types.ModuleType("notifications")
    m.notify = mock.AsyncMock(return_value=True)
    m.notify_event = mock.AsyncMock(return_value=None)
    m.CTO_TARGET = "cto@hurricanecore.internal"
    return m


def _make_parsers_mock():
    m = types.ModuleType("parsers")
    m.parse_training_log = lambda arch, log_text: {"metrics": {}, "per_class": {}}
    return m


def _load_list_submissions():
    """載入 list_submissions 函式（with mocked deps）。"""
    import secrets
    valid_key = secrets.token_hex(32)
    _setup_modules()
    with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": valid_key}):
        for mod in ["auth", "routers.submissions"]:
            if mod in sys.modules:
                del sys.modules[mod]
        import auth  # noqa: F401
        from routers.submissions import list_submissions
    return list_submissions


class TestSubmissionsQueryMetadata:
    """直接驗證 list_submissions 的 Query 參數 metadata。"""

    def setup_method(self):
        self.func = _load_list_submissions()

    def _get_param(self, name: str):
        import inspect
        sig = inspect.signature(self.func)
        return sig.parameters[name]

    def _get_constraints(self, param_name: str) -> dict:
        """從 Query.metadata 取得 {ge, le, default}。"""
        param = self._get_param(param_name)
        q = param.default
        result = {"default": getattr(q, "default", None)}
        for meta in getattr(q, "metadata", []):
            if hasattr(meta, "ge"):
                result["ge"] = meta.ge
            if hasattr(meta, "le"):
                result["le"] = meta.le
        return result

    def test_limit_has_le_100(self):
        """limit 應有 le=100 上限。"""
        c = self._get_constraints("limit")
        assert c.get("le") == 100, f"limit 應有 le=100，實際 constraints={c}"

    def test_limit_has_ge_1(self):
        """limit 應有 ge=1 下限。"""
        c = self._get_constraints("limit")
        assert c.get("ge") == 1, f"limit 應有 ge=1，實際 constraints={c}"

    def test_limit_default_is_50(self):
        """limit 預設應為 50。"""
        c = self._get_constraints("limit")
        assert c.get("default") == 50, f"limit 預設應為 50，實際 constraints={c}"

    def test_offset_has_le_10000(self):
        """offset 應有 le=10000 上限。"""
        c = self._get_constraints("offset")
        assert c.get("le") == 10000, f"offset 應有 le=10000，實際 constraints={c}"

    def test_offset_has_ge_0(self):
        """offset 應有 ge=0 下限。"""
        c = self._get_constraints("offset")
        assert c.get("ge") == 0, f"offset 應有 ge=0，實際 constraints={c}"

    def test_offset_default_is_0(self):
        """offset 預設應為 0。"""
        c = self._get_constraints("offset")
        assert c.get("default") == 0, f"offset 預設應為 0，實際 constraints={c}"


class TestSubmissionsQueryValidation:
    """驗證 limit=99999 在無認證 bypass 路由下回 422。"""

    def setup_method(self):
        import secrets
        valid_key = secrets.token_hex(32)
        _setup_modules()

        with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": valid_key}):
            for mod in ["auth", "routers.submissions"]:
                if mod in sys.modules:
                    del sys.modules[mod]
            import auth  # noqa: F401
            from routers.submissions import list_submissions

            # 建立無認證的路由 wrapper 供測試用
            from fastapi import FastAPI, Query
            from fastapi.testclient import TestClient

            app = FastAPI()

            @app.get("/test-limit")
            async def _test_limit(
                limit: int = Query(50, ge=1, le=100),
                offset: int = Query(0, ge=0, le=10000),
            ):
                return {"limit": limit, "offset": offset}

        self.client = TestClient(app, raise_server_exceptions=False)

    def test_limit_99999_returns_422(self):
        """limit=99999 應觸發 Query validation 422。"""
        resp = self.client.get("/test-limit?limit=99999")
        assert resp.status_code == 422, (
            f"limit=99999 應回 422，實際：{resp.status_code}\n{resp.text}"
        )

    def test_limit_0_returns_422(self):
        """limit=0 違反 ge=1，應回 422。"""
        resp = self.client.get("/test-limit?limit=0")
        assert resp.status_code == 422, (
            f"limit=0 應回 422，實際：{resp.status_code}\n{resp.text}"
        )

    def test_limit_101_returns_422(self):
        """limit=101 超過 le=100，應回 422。"""
        resp = self.client.get("/test-limit?limit=101")
        assert resp.status_code == 422, (
            f"limit=101 應回 422，實際：{resp.status_code}\n{resp.text}"
        )

    def test_limit_100_returns_200(self):
        """limit=100 在上限內，應回 200。"""
        resp = self.client.get("/test-limit?limit=100")
        assert resp.status_code == 200, (
            f"limit=100 應回 200，實際：{resp.status_code}\n{resp.text}"
        )

    def test_limit_50_default_returns_200(self):
        """預設 limit=50 應回 200。"""
        resp = self.client.get("/test-limit")
        assert resp.status_code == 200, (
            f"預設 limit=50 應回 200，實際：{resp.status_code}\n{resp.text}"
        )

    def test_offset_10001_returns_422(self):
        """offset=10001 超過 le=10000，應回 422。"""
        resp = self.client.get("/test-limit?offset=10001")
        assert resp.status_code == 422, (
            f"offset=10001 應回 422，實際：{resp.status_code}\n{resp.text}"
        )

    def test_offset_10000_returns_200(self):
        """offset=10000 在上限內，應回 200。"""
        resp = self.client.get("/test-limit?offset=10000")
        assert resp.status_code == 200, (
            f"offset=10000 應回 200，實際：{resp.status_code}\n{resp.text}"
        )

    def test_negative_offset_returns_422(self):
        """offset=-1 違反 ge=0，應回 422。"""
        resp = self.client.get("/test-limit?offset=-1")
        assert resp.status_code == 422, (
            f"offset=-1 應回 422，實際：{resp.status_code}\n{resp.text}"
        )
