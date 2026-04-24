"""
test_integration_auth.py — auth 模組整合測試

測試重點：
1. _verify_lids_token：cache 命中不重複打 LIDS（P1-2）
2. verify_api_key：DB → bootstrap fallback 順序
3. assert_role / require_role：role check 白名單
4. 已知不安全預設 key 被拒絕
"""

import sys
import types
import unittest.mock as mock
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# 設定 mock（不需要真實 DB）
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("SKIP_ROLE_CHECK", "false")   # auth 測試需要真正測 role check

# 確保 models 被 mock（auth.py 的 _verify_api_key_db 會 import models）
if "models" not in sys.modules:
    m = mock.MagicMock()
    m.SessionLocal = mock.MagicMock()
    m.ApiKey = mock.MagicMock()
    sys.modules["models"] = m

# 清除可能已存在的 auth 快取
if "auth" in sys.modules:
    del sys.modules["auth"]

import auth
from cachetools import TTLCache


# ---------------------------------------------------------------------------
# P1-2: TTLCache 命中測試
# ---------------------------------------------------------------------------

class TestTokenCache:

    def setup_method(self):
        """每個測試前清空 cache"""
        auth._TOKEN_CACHE.clear()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_lids(self):
        """首次呼叫 → 打 LIDS"""
        fake_userinfo = {"sub": "user-123", "preferred_username": "alice"}

        with mock.patch("auth.httpx.AsyncClient") as mock_client_cls:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = fake_userinfo

            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = mock.AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await auth._verify_lids_token("test-token-abc")

        assert result == fake_userinfo
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_lids(self):
        """第二次呼叫同一 token → 不打 LIDS（cache 命中）"""
        fake_userinfo = {"sub": "user-456", "preferred_username": "bob"}
        token = "cached-token-xyz"

        call_count = 0

        async def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = mock.MagicMock()
            resp.status_code = 200
            resp.json.return_value = fake_userinfo
            return resp

        with mock.patch("auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.get = fake_get
            mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = mock.AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            r1 = await auth._verify_lids_token(token)
            r2 = await auth._verify_lids_token(token)
            r3 = await auth._verify_lids_token(token)

        assert r1 == fake_userinfo
        assert r2 == fake_userinfo
        assert r3 == fake_userinfo
        # LIDS 只被呼叫一次
        assert call_count == 1, f"LIDS 應只被呼叫 1 次，實際呼叫 {call_count} 次"

    @pytest.mark.asyncio
    async def test_different_tokens_separate_cache_entries(self):
        """不同 token → 各自打 LIDS，各自快取"""
        call_count = 0

        async def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = mock.MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"sub": f"user-{call_count}"}
            return resp

        with mock.patch("auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock.AsyncMock()
            mock_client.get = fake_get
            mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = mock.AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await auth._verify_lids_token("token-A")
            await auth._verify_lids_token("token-B")
            # 第三次用 token-A，應命中 cache
            await auth._verify_lids_token("token-A")

        assert call_count == 2, f"應只打 LIDS 2 次（各 token 首次），實際 {call_count} 次"

    @pytest.mark.asyncio
    async def test_lids_401_raises_http_exception(self):
        """LIDS 回 401 → 應拋 HTTPException(401)，且不快取"""
        from fastapi import HTTPException

        with mock.patch("auth.httpx.AsyncClient") as mock_client_cls:
            mock_response = mock.MagicMock()
            mock_response.status_code = 401

            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = mock.AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await auth._verify_lids_token("invalid-token")

        assert exc_info.value.status_code == 401
        # 確認沒有被快取
        import hashlib
        key = hashlib.sha256(b"invalid-token").hexdigest()[:32]
        assert key not in auth._TOKEN_CACHE


# ---------------------------------------------------------------------------
# API Key fallback 順序測試
# ---------------------------------------------------------------------------

class TestApiKeyFallback:

    def test_known_insecure_key_rejected(self):
        """已知不安全預設 key 被拒絕"""
        result = auth.verify_api_key("modelhub-dev-key-2026")
        assert result is None

    def test_empty_key_rejected(self):
        """空 key 被拒絕"""
        assert auth.verify_api_key("") is None
        assert auth.verify_api_key(None) is None

    def test_db_key_takes_priority(self):
        """DB key 優先於 bootstrap env key"""
        db_user = {"sub": "api_key:42", "name": "CI Pipeline"}

        with mock.patch("auth._verify_api_key_db", return_value=db_user):
            with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": "bootstrap-key"}):
                # 重新載入 module-level 變數
                auth._BOOTSTRAP_KEY_RAW = "bootstrap-key"
                result = auth.verify_api_key("some-db-key")

        assert result == db_user

    def test_bootstrap_key_fallback(self):
        """DB 查無 → fallback bootstrap env key"""
        with mock.patch("auth._verify_api_key_db", return_value=None):
            auth._BOOTSTRAP_KEY_RAW = "my-secure-bootstrap-key"
            result = auth.verify_api_key("my-secure-bootstrap-key")

        assert result is not None
        assert result["sub"] == "api_key:bootstrap"

    def test_invalid_key_returns_none(self):
        """DB 查無 + 不匹配 bootstrap → None"""
        with mock.patch("auth._verify_api_key_db", return_value=None):
            auth._BOOTSTRAP_KEY_RAW = "correct-key"
            result = auth.verify_api_key("wrong-key")

        assert result is None


# ---------------------------------------------------------------------------
# Role check 測試
# ---------------------------------------------------------------------------

class TestRoleCheck:

    def setup_method(self):
        # 確保 SKIP_ROLE_CHECK=false（測試結束後由 teardown 還原）
        self._original_skip = auth._SKIP_ROLE_CHECK
        auth._SKIP_ROLE_CHECK = False
        auth._ROLE_CLAIM_KEY = "modelhub_role"

    def teardown_method(self):
        # 還原 SKIP_ROLE_CHECK，不污染其他測試
        auth._SKIP_ROLE_CHECK = self._original_skip

    def test_assert_role_pass(self):
        """user 有正確 role → 不拋例外"""
        user = {"sub": "user-1", "modelhub_role": "reviewer"}
        auth.assert_role("reviewer", user)  # 不應拋例外

    def test_assert_role_fail(self):
        """user role 不符 → HTTPException(403)"""
        from fastapi import HTTPException
        user = {"sub": "user-1", "modelhub_role": "viewer"}
        with pytest.raises(HTTPException) as exc_info:
            auth.assert_role("reviewer", user)
        assert exc_info.value.status_code == 403

    def test_assert_role_api_key_bypasses(self):
        """API Key 使用者（sub 以 api_key: 開頭）略過 role check"""
        user = {"sub": "api_key:123", "name": "service"}
        auth.assert_role("reviewer", user)  # 不應拋例外

    def test_assert_role_skip_when_env_set(self):
        """SKIP_ROLE_CHECK=true → 無論 role 都通過"""
        auth._SKIP_ROLE_CHECK = True
        user = {"sub": "user-1", "modelhub_role": "nobody"}
        auth.assert_role("reviewer", user)  # 不應拋例外
        auth._SKIP_ROLE_CHECK = False  # 還原

    def test_assert_role_none_user(self):
        """user 為 None/空 dict → HTTPException(403)"""
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            auth.assert_role("reviewer", {})
