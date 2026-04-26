"""
test_kaggle_poller_status_update.py — 驗證 Kaggle Poller 狀態更新邏輯

修復工單：MH-2026-019 / poller bug
問題：
  1. _kaggle_env_ready() 只看 env var，忽略 ~/.kaggle/kaggle.json
     → 導致 KAGGLE_USERNAME/KEY 未設時 poll_once() 永遠 early return
  2. poll_once() 缺少 debug log 無法診斷狀態推進
"""

import sys
import os
import json
import types
import tempfile
import unittest.mock as mock
from unittest.mock import MagicMock, AsyncMock, patch, call
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_sub(req_no: str, kaggle_status: str = "queued", status: str = "training"):
    sub = MagicMock()
    sub.req_no = req_no
    sub.status = status
    sub.kaggle_status = kaggle_status
    sub.kaggle_kernel_slug = f"boardgamegroup/{req_no.lower()}"
    sub.arch = "yolov8m"
    sub.product = "test"
    sub.req_name = req_no
    sub.training_started_at = None
    sub.retry_count = 0
    sub.max_retries = 2
    sub.budget_exceeded_notified = False
    sub.max_budget_usd = 0
    sub.per_class_metrics = None
    sub.gpu_seconds = None
    sub.estimated_cost_usd = None
    sub.training_completed_at = None
    sub.kaggle_status_updated_at = None
    return sub


# ---------------------------------------------------------------------------
# 1. _kaggle_env_ready — 從 kaggle.json 載入 credentials
# ---------------------------------------------------------------------------

class TestKaggleEnvReady:

    def test_ready_via_env_var(self, monkeypatch):
        """KAGGLE_USERNAME + KAGGLE_KEY 設定時應回傳 True"""
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setenv("KAGGLE_KEY", "testkey123")

        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        assert kp._kaggle_env_ready() is True

    def test_ready_via_kaggle_json(self, monkeypatch, tmp_path):
        """env var 未設但 kaggle.json 存在時應自動載入並回傳 True"""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        # 建立 fake kaggle.json
        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(json.dumps({"username": "boardgamegroup", "key": "fakekey"}))

        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with patch("pollers.kaggle_poller.Path") as mock_path_cls:
            # 讓 Path(expanduser("~/.kaggle/kaggle.json")) 指向 tmp_path
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path_inst.read_text.return_value = json.dumps({"username": "boardgamegroup", "key": "fakekey"})
            mock_path_cls.return_value = mock_path_inst

            result = kp._kaggle_env_ready()

        assert result is True
        # env var 應被注入
        assert os.environ.get("KAGGLE_USERNAME") == "boardgamegroup"
        assert os.environ.get("KAGGLE_KEY") == "fakekey"

    def test_not_ready_when_no_env_no_json(self, monkeypatch):
        """env var 未設且 kaggle.json 不存在時應回傳 False"""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with patch("pollers.kaggle_poller.Path") as mock_path_cls:
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = False
            mock_path_cls.return_value = mock_path_inst

            result = kp._kaggle_env_ready()

        assert result is False

    def test_not_ready_when_json_malformed(self, monkeypatch):
        """kaggle.json 格式錯誤時不應 crash，應回傳 False"""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with patch("pollers.kaggle_poller.Path") as mock_path_cls:
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path_inst.read_text.return_value = "not-valid-json{"
            mock_path_cls.return_value = mock_path_inst

            result = kp._kaggle_env_ready()

        assert result is False


# ---------------------------------------------------------------------------
# 2. poll_once — kaggle_status queued → running 正確更新
# ---------------------------------------------------------------------------

class TestPollOnceStatusUpdate:

    @pytest.mark.asyncio
    async def test_queued_to_running_updates_db(self, monkeypatch):
        """
        DB kaggle_status=queued，Kaggle API 回 running
        → kaggle_status 應更新為 running，history 新增一筆
        """
        monkeypatch.setenv("KAGGLE_USERNAME", "boardgamegroup")
        monkeypatch.setenv("KAGGLE_KEY", "fakekey")

        sub = _make_sub("MH-2026-018", kaggle_status="queued")
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [sub]

        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with patch.object(kp, "_get_kaggle_api", return_value=None), \
             patch.object(kp, "_fetch_kernel_status",
                          new=AsyncMock(return_value={"status": "running", "raw": "has status running"})), \
             patch.object(kp, "_check_overtime", new=AsyncMock()), \
             patch.object(kp, "_check_budget", new=AsyncMock()), \
             patch.object(kp, "_check_quota_warning", new=AsyncMock()), \
             patch.object(kp, "SessionLocal", return_value=db):

            result = await kp.poll_once()

        assert result["checked"] == 1
        assert result["changed"] == 1
        assert sub.kaggle_status == "running"
        db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_queued_to_error_triggers_on_kernel_error(self, monkeypatch):
        """
        DB kaggle_status=queued，Kaggle API 回 error
        → kaggle_status 更新 + _on_kernel_error 被呼叫
        """
        monkeypatch.setenv("KAGGLE_USERNAME", "boardgamegroup")
        monkeypatch.setenv("KAGGLE_KEY", "fakekey")

        sub = _make_sub("MH-2026-019", kaggle_status="queued")
        sub.retry_count = 2  # 讓 _on_kernel_error 走真失敗路徑
        sub.max_retries = 2
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [sub]

        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        on_error_called = []

        async def fake_on_error(db_, sub_, raw_):
            on_error_called.append((sub_.req_no, raw_))

        with patch.object(kp, "_get_kaggle_api", return_value=None), \
             patch.object(kp, "_fetch_kernel_status",
                          new=AsyncMock(return_value={"status": "error", "raw": "kernel error"})), \
             patch.object(kp, "_check_overtime", new=AsyncMock()), \
             patch.object(kp, "_check_budget", new=AsyncMock()), \
             patch.object(kp, "_check_quota_warning", new=AsyncMock()), \
             patch.object(kp, "_on_kernel_error", new=fake_on_error), \
             patch.object(kp, "SessionLocal", return_value=db):

            result = await kp.poll_once()

        assert result["error"] == 1
        assert len(on_error_called) == 1
        assert on_error_called[0][0] == "MH-2026-019"

    @pytest.mark.asyncio
    async def test_poll_skipped_when_no_credentials(self, monkeypatch):
        """
        env var 未設且 kaggle.json 不存在時 poll_once 應 early return
        """
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with patch("pollers.kaggle_poller.Path") as mock_path_cls:
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = False
            mock_path_cls.return_value = mock_path_inst

            result = await kp.poll_once()

        assert result == {"skipped": True}

    @pytest.mark.asyncio
    async def test_same_status_no_commit(self, monkeypatch):
        """
        DB kaggle_status 已是 running，Kaggle API 也回 running
        → changed=0，不多餘 commit history
        """
        monkeypatch.setenv("KAGGLE_USERNAME", "boardgamegroup")
        monkeypatch.setenv("KAGGLE_KEY", "fakekey")

        sub = _make_sub("MH-2026-018", kaggle_status="running")
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [sub]

        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with patch.object(kp, "_get_kaggle_api", return_value=None), \
             patch.object(kp, "_fetch_kernel_status",
                          new=AsyncMock(return_value={"status": "running", "raw": "running"})), \
             patch.object(kp, "_check_overtime", new=AsyncMock()), \
             patch.object(kp, "_check_budget", new=AsyncMock()), \
             patch.object(kp, "_check_quota_warning", new=AsyncMock()), \
             patch.object(kp, "SessionLocal", return_value=db):

            result = await kp.poll_once()

        assert result["changed"] == 0


# ---------------------------------------------------------------------------
# 3. _fetch_kernel_status — Kaggle CLI 輸出含 Warning 行仍能解析
# ---------------------------------------------------------------------------

class TestFetchKernelStatusParsing:

    @pytest.mark.asyncio
    async def test_parses_status_with_warning_prefix(self):
        """
        Kaggle CLI stdout 可能含 Warning 行：
          'Warning: ...version...\nslug has status "running"\n'
        regex 應正確抓到 "running"
        """
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        stdout_with_warning = (
            'Warning: Looks like you\'re using an outdated API Version, '
            'please consider updating (server 2.0.2 / client 1.6.17)\n'
            'boardgamegroup/mh-2026-018-ppe-detection-yolov8m has status "running"\n'
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(
            stdout_with_warning.encode(), b""
        ))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("shutil.which", return_value="/usr/local/bin/kaggle"):
            result = await kp._fetch_kernel_status(None, "boardgamegroup/mh-2026-018")

        assert result is not None
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_parses_error_status(self):
        """Kaggle API 回 error 狀態應能正確解析"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        stdout = 'boardgamegroup/mh-2026-019 has status "error"\n'
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("shutil.which", return_value="/usr/local/bin/kaggle"):
            result = await kp._fetch_kernel_status(None, "boardgamegroup/mh-2026-019")

        assert result is not None
        assert result["status"] == "error"
