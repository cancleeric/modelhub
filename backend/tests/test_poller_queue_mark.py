"""
test_poller_queue_mark.py — 驗證 Kaggle/Lightning poller 在完成與失敗時
正確呼叫 QueueManager.mark_done_by_req / mark_failed_by_req。

修復工單：dispatcher 靜默鎖死 bug（count_running() 永遠 >= MAX_CONCURRENT_TRAININGS）。
"""

import sys
import os
import types
import unittest.mock as mock
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_submission(req_no: str, status: str = "training") -> MagicMock:
    sub = MagicMock()
    sub.req_no = req_no
    sub.status = status
    sub.kaggle_status = "running"
    sub.kaggle_kernel_slug = f"boardgamegroup/{req_no.lower()}"
    sub.lightning_studio_name = None
    sub.arch = "yolov8m"
    sub.product = "test-product"
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
# Kaggle poller — _on_kernel_complete 呼叫 mark_done_by_req
# ---------------------------------------------------------------------------

class TestKagglePollerQueueMarkDone:
    """_on_kernel_complete 完成後必須呼叫 QueueManager.mark_done_by_req"""

    @pytest.mark.asyncio
    async def test_on_kernel_complete_calls_mark_done(self):
        sub = _make_submission("MH-2026-TEST-001", status="training")

        db = MagicMock()
        db2 = MagicMock()

        # mock SessionLocal 回傳 db2
        mock_session_local = MagicMock(return_value=db2)

        # mock QueueManager
        mock_qm = MagicMock()
        mock_qm.mark_done_by_req = MagicMock()

        with patch.dict("sys.modules", {
            "queue_manager": types.SimpleNamespace(QueueManager=mock_qm),
        }):
            import importlib
            import pollers.kaggle_poller as kp
            # patch SessionLocal + _download_kernel_output + notify_event + _next_version_for
            with patch.object(kp, "SessionLocal", mock_session_local), \
                 patch.object(kp, "_download_kernel_output", new=AsyncMock(return_value=None)), \
                 patch.object(kp, "notify_event", new=AsyncMock(return_value=None)), \
                 patch.object(kp, "_next_version_for", return_value="v1"):

                # _on_kernel_complete 中會 import queue_manager，需 patch sys.modules
                with patch.dict("sys.modules", {"queue_manager": types.SimpleNamespace(QueueManager=mock_qm)}):
                    await kp._on_kernel_complete(db, sub)

        mock_qm.mark_done_by_req.assert_called_once_with(db2, "MH-2026-TEST-001")
        db2.commit.assert_called_once()
        db2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_kernel_complete_idempotent_guard_skips_when_terminal(self):
        """已在終態（trained）時不應執行任何邏輯（幂等 guard）"""
        sub = _make_submission("MH-2026-TEST-002", status="trained")

        db = MagicMock()
        mock_qm = MagicMock()

        with patch.dict("sys.modules", {"queue_manager": types.SimpleNamespace(QueueManager=mock_qm)}):
            import pollers.kaggle_poller as kp
            await kp._on_kernel_complete(db, sub)

        # 幂等 guard 應跳過，mark_done_by_req 不被呼叫
        mock_qm.mark_done_by_req.assert_not_called()


# ---------------------------------------------------------------------------
# Kaggle poller — _on_kernel_error 達上限時呼叫 mark_failed_by_req
# ---------------------------------------------------------------------------

class TestKagglePollerQueueMarkFailed:
    """_on_kernel_error 達重試上限（真失敗）時必須呼叫 mark_failed_by_req"""

    @pytest.mark.asyncio
    async def test_on_kernel_error_final_failure_calls_mark_failed(self):
        sub = _make_submission("MH-2026-TEST-003", status="training")
        sub.retry_count = 2  # 已達上限
        sub.max_retries = 2

        db = MagicMock()
        db2 = MagicMock()
        mock_session_local = MagicMock(return_value=db2)
        mock_qm = MagicMock()
        mock_qm.mark_failed_by_req = MagicMock()

        with patch.dict("sys.modules", {"queue_manager": types.SimpleNamespace(QueueManager=mock_qm)}):
            import pollers.kaggle_poller as kp
            with patch.object(kp, "SessionLocal", mock_session_local), \
                 patch.object(kp, "notify_event", new=AsyncMock(return_value=None)), \
                 patch.object(kp, "_append_training_failed_summary", MagicMock()):

                with patch.dict("sys.modules", {"queue_manager": types.SimpleNamespace(QueueManager=mock_qm)}):
                    await kp._on_kernel_error(db, sub, "kernel crashed")

        mock_qm.mark_failed_by_req.assert_called_once_with(db2, "MH-2026-TEST-003", "kaggle training error")
        db2.commit.assert_called_once()
        db2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_kernel_error_retry_path_does_not_call_mark_failed(self):
        """自動重試路徑（尚未達上限）不應呼叫 mark_failed_by_req"""
        sub = _make_submission("MH-2026-TEST-004", status="training")
        sub.retry_count = 0
        sub.max_retries = 2

        db = MagicMock()
        mock_qm = MagicMock()

        with patch.dict("sys.modules", {"queue_manager": types.SimpleNamespace(QueueManager=mock_qm)}):
            import pollers.kaggle_poller as kp
            with patch.object(kp, "_push_kernel", new=AsyncMock(return_value=True)), \
                 patch.object(kp, "notify_event", new=AsyncMock(return_value=None)):

                with patch.dict("sys.modules", {"queue_manager": types.SimpleNamespace(QueueManager=mock_qm)}):
                    await kp._on_kernel_error(db, sub, "transient error")

        mock_qm.mark_failed_by_req.assert_not_called()


# ---------------------------------------------------------------------------
# Lightning poller — _process_submission complete 呼叫 mark_done_by_req
# ---------------------------------------------------------------------------

class TestLightningPollerQueueMarkDone:
    """_process_submission status=complete 時必須呼叫 mark_done_by_req"""

    def test_process_submission_complete_calls_mark_done(self):
        sub = _make_submission("MH-2026-TEST-005", status="training")
        sub.lightning_studio_name = "mh-2026-test-005"

        db = MagicMock()
        db2 = MagicMock()
        mock_session_local = MagicMock(return_value=db2)
        mock_qm = MagicMock()
        mock_qm.mark_done_by_req = MagicMock()

        import pollers.lightning_poller as lp

        with patch.object(lp, "_fetch_job_status", return_value={"status": "complete", "raw": "stopped"}), \
             patch.object(lp, "_download_job_output", return_value=None), \
             patch.object(lp, "SessionLocal", mock_session_local), \
             patch.object(lp, "notify_event", AsyncMock(return_value=None)), \
             patch.object(lp, "_next_version_for", return_value="v1"):

            with patch.dict("sys.modules", {"queue_manager": types.SimpleNamespace(QueueManager=mock_qm)}):
                lp._process_submission(db, sub)

        mock_qm.mark_done_by_req.assert_called_once_with(db2, "MH-2026-TEST-005")
        db2.commit.assert_called_once()
        db2.close.assert_called_once()


# ---------------------------------------------------------------------------
# Lightning poller — _process_submission error 呼叫 mark_failed_by_req
# ---------------------------------------------------------------------------

class TestLightningPollerQueueMarkFailed:
    """_process_submission status=error 時必須呼叫 mark_failed_by_req"""

    def test_process_submission_error_calls_mark_failed(self):
        sub = _make_submission("MH-2026-TEST-006", status="training")
        sub.lightning_studio_name = "mh-2026-test-006"

        db = MagicMock()
        db2 = MagicMock()
        mock_session_local = MagicMock(return_value=db2)
        mock_qm = MagicMock()
        mock_qm.mark_failed_by_req = MagicMock()

        import pollers.lightning_poller as lp

        with patch.object(lp, "_fetch_job_status", return_value={"status": "error", "raw": "failed"}), \
             patch.object(lp, "SessionLocal", mock_session_local), \
             patch.object(lp, "notify_event", AsyncMock(return_value=None)), \
             patch.object(lp, "_append_training_failed_summary_lightning", MagicMock()):

            with patch.dict("sys.modules", {"queue_manager": types.SimpleNamespace(QueueManager=mock_qm)}):
                lp._process_submission(db, sub)

        mock_qm.mark_failed_by_req.assert_called_once_with(db2, "MH-2026-TEST-006", "lightning training error")
        db2.commit.assert_called_once()
        db2.close.assert_called_once()
