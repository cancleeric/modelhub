"""
test_auth_security.py — M14 S6：bootstrap key 安全性測試

驗收標準：
- 短 key（長度 < 32）觸發啟動拒絕（RuntimeError）
- 已知不安全 key 清單測試（mh-cto-ops-2026-0418 等）
- 含不安全 pattern（dev/ops/2026）拒絕啟動
- 合法 64-char hex key 通過
"""
import importlib
import os
import sys
import types
import unittest.mock as mock

import pytest


def _reload_auth_with_key(key: str | None) -> types.ModuleType:
    """以指定 MODELHUB_API_KEY 重新載入 auth 模組。"""
    env_patch = {} if key is None else {"MODELHUB_API_KEY": key}
    # 確保 models mock 存在
    if "models" not in sys.modules:
        models_mock = mock.MagicMock()
        sys.modules["models"] = models_mock

    with mock.patch.dict(os.environ, env_patch, clear=(key is None)):
        # 若 key is None，需把 MODELHUB_API_KEY 從 env 移除
        if key is None:
            os.environ.pop("MODELHUB_API_KEY", None)
        if "auth" in sys.modules:
            del sys.modules["auth"]
        import auth  # noqa: PLC0415
        return auth


class TestBootstrapKeySecurity:
    """S4: _check_bootstrap_key_security 啟動時驗證。"""

    def test_known_insecure_key_mh_cto_ops_2026_0418(self):
        """已知不安全 key mh-cto-ops-2026-0418 必須觸發 RuntimeError。"""
        if "auth" in sys.modules:
            del sys.modules["auth"]
        with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": "mh-cto-ops-2026-0418"}):
            with pytest.raises(RuntimeError, match="insecure bootstrap key detected"):
                import auth  # noqa: PLC0415

    def test_known_insecure_key_mh_cto_ops_2026(self):
        """已知不安全 key mh-cto-ops-2026 必須觸發 RuntimeError。"""
        if "auth" in sys.modules:
            del sys.modules["auth"]
        with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": "mh-cto-ops-2026"}):
            with pytest.raises(RuntimeError, match="insecure bootstrap key detected"):
                import auth  # noqa: PLC0415

    def test_known_insecure_key_modelhub_dev_key(self):
        """已知不安全 key modelhub-dev-key-2026 必須觸發 RuntimeError。"""
        if "auth" in sys.modules:
            del sys.modules["auth"]
        with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": "modelhub-dev-key-2026"}):
            with pytest.raises(RuntimeError, match="insecure bootstrap key detected"):
                import auth  # noqa: PLC0415

    def test_short_key_rejected(self):
        """長度 < 32 的 key 必須觸發 RuntimeError。"""
        short_key = "a" * 31
        if "auth" in sys.modules:
            del sys.modules["auth"]
        with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": short_key}):
            with pytest.raises(RuntimeError, match="insecure bootstrap key detected"):
                import auth  # noqa: PLC0415

    def test_key_with_dev_pattern_rejected(self):
        """含 'dev' pattern 的 key 必須觸發 RuntimeError（即使長度夠）。"""
        bad_key = "a" * 28 + "dev0"  # 32 chars, contains 'dev'
        if "auth" in sys.modules:
            del sys.modules["auth"]
        with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": bad_key}):
            with pytest.raises(RuntimeError, match="insecure bootstrap key detected"):
                import auth  # noqa: PLC0415

    def test_key_with_2026_pattern_rejected(self):
        """含 '2026' pattern 的 key 必須觸發 RuntimeError。"""
        bad_key = "a" * 28 + "2026"  # 32 chars, contains '2026'
        if "auth" in sys.modules:
            del sys.modules["auth"]
        with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": bad_key}):
            with pytest.raises(RuntimeError, match="insecure bootstrap key detected"):
                import auth  # noqa: PLC0415

    def test_valid_64char_hex_key_passes(self):
        """合法 64-char hex key 應正常載入，不拋 RuntimeError。"""
        import secrets
        valid_key = secrets.token_hex(32)  # 64 chars, no insecure patterns
        if "auth" in sys.modules:
            del sys.modules["auth"]
        with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": valid_key}):
            import auth  # noqa: PLC0415
            assert auth._BOOTSTRAP_KEY_RAW == valid_key

    def test_no_key_does_not_raise(self):
        """未設定 MODELHUB_API_KEY 不應拋 RuntimeError（僅 warning，DB-backed key 仍可用）。"""
        if "auth" in sys.modules:
            del sys.modules["auth"]
        env = {k: v for k, v in os.environ.items() if k != "MODELHUB_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            import auth  # noqa: PLC0415
            assert auth._BOOTSTRAP_KEY_RAW is None

    def test_verify_api_key_rejects_known_insecure(self):
        """verify_api_key 應拒絕已知不安全 key（即使繞過啟動檢查也拒絕）。"""
        import secrets
        valid_key = secrets.token_hex(32)
        if "auth" in sys.modules:
            del sys.modules["auth"]
        with mock.patch.dict(os.environ, {"MODELHUB_API_KEY": valid_key}):
            import auth  # noqa: PLC0415
            # 模擬用舊的已知不安全 key 呼叫
            result = auth.verify_api_key("modelhub-dev-key-2026")
            assert result is None
