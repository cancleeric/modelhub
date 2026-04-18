"""
Sprint 22 Task 22-3：LightningLauncher E2E 測試（mock SDK）

所有測試皆以 unittest.mock patch lightning_sdk，不實際連線 Lightning AI。
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def launcher_with_creds(monkeypatch):
    """設定有效 LIGHTNING credentials 的 LightningLauncher"""
    monkeypatch.setenv("LIGHTNING_USER_ID", "test-user-id")
    monkeypatch.setenv("LIGHTNING_API_KEY", "test-api-key")
    monkeypatch.setenv("LIGHTNING_USERNAME", "testuser")
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "test-teamspace")

    # Re-import to pick up env changes（module-level constants）
    import importlib
    import resources.lightning_launcher as ll_mod
    importlib.reload(ll_mod)
    return ll_mod.LightningLauncher()


@pytest.fixture
def launcher_no_creds(monkeypatch):
    """沒有 credentials 的 LightningLauncher"""
    monkeypatch.delenv("LIGHTNING_USER_ID", raising=False)
    monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)

    import importlib
    import resources.lightning_launcher as ll_mod
    importlib.reload(ll_mod)
    return ll_mod.LightningLauncher()


# ---------------------------------------------------------------------------
# submit_job
# ---------------------------------------------------------------------------

class TestSubmitJob:
    def test_submit_job_no_creds_returns_failure(self, launcher_no_creds):
        result = launcher_no_creds.submit_job("MH-2026-001", "/tmp/dataset")
        assert result["success"] is False
        assert "LIGHTNING_USER_ID" in result["reason"] or "LIGHTNING_API_KEY" in result["reason"]

    def test_submit_job_success(self, launcher_with_creds):
        mock_studio = MagicMock()
        mock_machine = MagicMock()
        mock_machine.T4 = "T4"

        # lightning_launcher 是 lazy import，需 patch lightning_sdk 模組本身
        with patch.dict("sys.modules", {
            "lightning_sdk": MagicMock(Studio=MagicMock(return_value=mock_studio), Machine=mock_machine),
        }):
            result = launcher_with_creds.submit_job(
                req_no="MH-2026-001",
                dataset_path="/tmp/nonexistent_dataset",
                config={"epochs": 10},
            )

        assert result["success"] is True
        assert "studio_name" in result
        assert result["studio_name"] == "mh-2026-001"

    def test_submit_job_sdk_exception_returns_failure(self, launcher_with_creds):
        mock_sdk = MagicMock()
        mock_sdk.Studio.side_effect = RuntimeError("SDK error")
        with patch.dict("sys.modules", {"lightning_sdk": mock_sdk}):
            result = launcher_with_creds.submit_job("MH-2026-002", "/tmp/dataset")
        assert result["success"] is False
        assert "SDK error" in result["reason"]

    def test_submit_job_req_no_normalised_to_studio_name(self, launcher_with_creds):
        mock_studio = MagicMock()
        mock_machine = MagicMock()
        mock_machine.T4 = "T4"
        with patch.dict("sys.modules", {
            "lightning_sdk": MagicMock(Studio=MagicMock(return_value=mock_studio), Machine=mock_machine),
        }):
            result = launcher_with_creds.submit_job("MH_2026_003", "/tmp/ds")
        # 底線轉 dash，大寫轉小寫
        assert result.get("studio_name") == "mh-2026-003"


# ---------------------------------------------------------------------------
# get_job_status
# ---------------------------------------------------------------------------

def _make_sdk_mock(studio_status_str: str):
    """建立 lightning_sdk mock，讓 studio.status 等於給定字串"""
    mock_status_enum = MagicMock()
    mock_status_enum.Running = "Running"
    mock_status_enum.Stopped = "Stopped"
    mock_status_enum.Completed = "Completed"
    mock_status_enum.Failed = "Failed"
    mock_status_enum.NotCreated = "NotCreated"

    mock_studio = MagicMock()
    mock_studio.status = studio_status_str

    mock_sdk = MagicMock()
    mock_sdk.Studio = MagicMock(return_value=mock_studio)
    mock_sdk.Status = mock_status_enum
    return mock_sdk, mock_studio


class TestGetJobStatus:
    def test_get_job_status_no_creds_returns_error(self, launcher_no_creds):
        assert launcher_no_creds.get_job_status("some-studio") == "error"

    def test_get_job_status_running(self, launcher_with_creds):
        mock_sdk, _ = _make_sdk_mock("Running")
        with patch.dict("sys.modules", {"lightning_sdk": mock_sdk}):
            result = launcher_with_creds.get_job_status("test-studio")
        assert result == "running"

    def test_get_job_status_stopped_maps_to_complete(self, launcher_with_creds):
        mock_sdk, _ = _make_sdk_mock("Stopped")
        with patch.dict("sys.modules", {"lightning_sdk": mock_sdk}):
            result = launcher_with_creds.get_job_status("test-studio")
        assert result == "complete"

    def test_get_job_status_failed_maps_to_error(self, launcher_with_creds):
        mock_sdk, _ = _make_sdk_mock("Failed")
        with patch.dict("sys.modules", {"lightning_sdk": mock_sdk}):
            result = launcher_with_creds.get_job_status("test-studio")
        assert result == "error"

    def test_get_job_status_exception_returns_error(self, launcher_with_creds):
        mock_sdk = MagicMock()
        mock_sdk.Studio.side_effect = Exception("network error")
        with patch.dict("sys.modules", {"lightning_sdk": mock_sdk}):
            result = launcher_with_creds.get_job_status("test-studio")
        assert result == "error"

    @pytest.mark.parametrize("sdk_status,expected", [
        ("Running",    "running"),
        ("Stopped",    "complete"),
        ("Completed",  "complete"),
        ("Failed",     "error"),
        ("NotCreated", "unknown"),
    ])
    def test_get_job_status_all_status_values(self, launcher_with_creds, sdk_status, expected):
        mock_sdk, _ = _make_sdk_mock(sdk_status)
        with patch.dict("sys.modules", {"lightning_sdk": mock_sdk}):
            result = launcher_with_creds.get_job_status("test-studio")
        assert result == expected


# ---------------------------------------------------------------------------
# download_model
# ---------------------------------------------------------------------------

class TestDownloadModel:
    def test_download_model_no_creds_returns_false(self, launcher_no_creds):
        assert launcher_no_creds.download_model("test-studio", "/tmp/best.pt") is False

    def test_download_model_best_pt_not_found(self, launcher_with_creds, tmp_path):
        mock_studio = MagicMock()
        mock_studio.run.return_value = ""  # 空字串 → best.pt 找不到

        mock_sdk = MagicMock()
        mock_sdk.Studio = MagicMock(return_value=mock_studio)
        with patch.dict("sys.modules", {"lightning_sdk": mock_sdk}):
            result = launcher_with_creds.download_model("test-studio", str(tmp_path / "best.pt"))

        assert result is False
        mock_studio.download_file.assert_not_called()

    def test_download_model_success(self, launcher_with_creds, tmp_path):
        mock_studio = MagicMock()
        mock_studio.run.return_value = "runs/train/exp/weights/best.pt"

        mock_sdk = MagicMock()
        mock_sdk.Studio = MagicMock(return_value=mock_studio)
        with patch.dict("sys.modules", {"lightning_sdk": mock_sdk}):
            result = launcher_with_creds.download_model("test-studio", str(tmp_path / "best.pt"))

        assert result is True
        mock_studio.download_file.assert_called_once()

    def test_download_model_exception_returns_false(self, launcher_with_creds, tmp_path):
        mock_sdk = MagicMock()
        mock_sdk.Studio.side_effect = Exception("network error")
        with patch.dict("sys.modules", {"lightning_sdk": mock_sdk}):
            result = launcher_with_creds.download_model("test-studio", str(tmp_path / "best.pt"))
        assert result is False


# ---------------------------------------------------------------------------
# poll_once（lightning_poller）
# ---------------------------------------------------------------------------

class TestLightningPollerPollOnce:
    def test_poll_once_skips_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)

        import importlib
        import pollers.lightning_poller as lp
        importlib.reload(lp)

        result = lp.poll_once()
        assert result.get("skipped") is True
        assert "LIGHTNING_API_KEY" in result.get("reason", "")
