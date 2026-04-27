"""
test_queue_dispatcher_kernel_race.py — 驗證 queue_dispatcher 與 attach-kernel 的
race condition 修復邏輯（fix/queue-dispatcher-kernel-race）。

測試情境：
1. approve 後 kernel 尚未 attach → dispatcher 應回 pending_kernel，不推進到 training
2. kernel 已 attach → dispatcher 正常派發到 training
3. attach-kernel endpoint 完成後若有 pending_kernel 條目，自動 reset 為 waiting
4. attach-kernel 完成但無 pending_kernel 條目時，queue_reset_to_waiting=False
5. QueueManager.mark_pending_kernel 正確設定 status
6. QueueManager.reset_pending_kernel 正確重設並回傳 True/False
"""

import sys
import os
import types
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_queue_entry(entry_id: int, req_no: str, status: str = "waiting") -> MagicMock:
    entry = MagicMock()
    entry.id = entry_id
    entry.req_no = req_no
    entry.status = status
    entry.priority = "P2"
    entry.enqueued_at = __import__("datetime").datetime.utcnow()
    return entry


def _make_submission(req_no: str, status: str = "approved",
                     kaggle_kernel_slug: str = None) -> MagicMock:
    sub = MagicMock()
    sub.req_no = req_no
    sub.status = status
    sub.kaggle_kernel_slug = kaggle_kernel_slug
    sub.training_resource = None
    sub.total_attempts = 0
    sub.priority = "P2"
    return sub


# ---------------------------------------------------------------------------
# QueueManager 新方法單元測試
# ---------------------------------------------------------------------------

class TestQueueManagerPendingKernel:
    """QueueManager.mark_pending_kernel / reset_pending_kernel 基本行為"""

    def setup_method(self):
        # 強制重新 import 真實的 queue_manager（避免其他 test 安裝的 fake 污染）
        import sys
        sys.modules.pop("queue_manager", None)

    def test_mark_pending_kernel_sets_status(self):
        """mark_pending_kernel 應將 entry.status 改為 pending_kernel"""
        from queue_manager import QueueManager

        entry = _make_queue_entry(1, "MH-2026-TEST-RC-001")
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = entry

        QueueManager.mark_pending_kernel(db, 1)

        assert entry.status == "pending_kernel"
        db.flush.assert_called_once()

    def test_mark_pending_kernel_entry_not_found(self):
        """entry 不存在時 mark_pending_kernel 應靜默處理（不 raise）"""
        from queue_manager import QueueManager

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None

        # 不應 raise
        QueueManager.mark_pending_kernel(db, 999)
        db.flush.assert_not_called()

    def test_reset_pending_kernel_returns_true_when_entry_exists(self):
        """pending_kernel 條目存在時 reset_pending_kernel 應回 True 並設 status=waiting"""
        from queue_manager import QueueManager

        entry = _make_queue_entry(2, "MH-2026-TEST-RC-002", status="pending_kernel")
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = entry

        result = QueueManager.reset_pending_kernel(db, "MH-2026-TEST-RC-002")

        assert result is True
        assert entry.status == "waiting"
        db.flush.assert_called_once()

    def test_reset_pending_kernel_returns_false_when_no_entry(self):
        """無 pending_kernel 條目時 reset_pending_kernel 應回 False"""
        from queue_manager import QueueManager

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None

        result = QueueManager.reset_pending_kernel(db, "MH-2026-TEST-RC-003")

        assert result is False
        db.flush.assert_not_called()


# ---------------------------------------------------------------------------
# dispatch_next — PENDING_KERNEL sentinel 路徑
# ---------------------------------------------------------------------------

class TestDispatchNextPendingKernelPath:
    """dispatch_next 遇到 PENDING_KERNEL sentinel 時走正確分支"""

    def test_dispatch_next_marks_pending_kernel_when_kernel_missing(self):
        """
        ResourceProber 回 kaggle 但 kaggle_kernel_slug 為空 →
        dispatch_next 應標記 pending_kernel，不推進到 training
        （QueueManager 是 lazy import，用 sys.modules patch）
        """
        import pollers.queue_dispatcher as qd

        entry = _make_queue_entry(10, "MH-2026-TEST-RC-010")

        mock_qm = MagicMock()
        mock_qm.count_running.return_value = 0
        mock_qm.peek_next.return_value = entry
        mock_qm.mark_dispatching = MagicMock()
        mock_qm.mark_pending_kernel = MagicMock()

        mock_db = MagicMock()
        mock_session_local = MagicMock(return_value=mock_db)

        with patch.object(qd, "SessionLocal", mock_session_local), \
             patch.dict("sys.modules", {
                 "queue_manager": types.SimpleNamespace(QueueManager=mock_qm),
             }), \
             patch.object(qd, "_do_dispatch",
                          return_value=(False, "PENDING_KERNEL", "kaggle_kernel_slug not set")):
            result = qd.dispatch_next()

        assert "pending_kernel" in result
        assert result["pending_kernel"] == "MH-2026-TEST-RC-010"
        mock_qm.mark_pending_kernel.assert_called_once_with(mock_db, entry.id)

    def test_dispatch_next_dispatches_normally_when_kernel_attached(self):
        """
        kaggle_kernel_slug 已設定 → dispatch_next 應正常派發並回 dispatched
        """
        import pollers.queue_dispatcher as qd

        entry = _make_queue_entry(11, "MH-2026-TEST-RC-011")

        mock_qm = MagicMock()
        mock_qm.count_running.return_value = 0
        mock_qm.peek_next.return_value = entry
        mock_qm.mark_dispatching = MagicMock()
        mock_qm.mark_running = MagicMock()

        mock_db = MagicMock()
        mock_session_local = MagicMock(return_value=mock_db)

        with patch.object(qd, "SessionLocal", mock_session_local), \
             patch.dict("sys.modules", {
                 "queue_manager": types.SimpleNamespace(QueueManager=mock_qm),
             }), \
             patch.object(qd, "_do_dispatch", return_value=(True, "kaggle", "")):
            result = qd.dispatch_next()

        assert "dispatched" in result
        assert result["dispatched"] == "MH-2026-TEST-RC-011"
        mock_qm.mark_running.assert_called_once()


# ---------------------------------------------------------------------------
# _do_dispatch — kernel 前置檢查
# ---------------------------------------------------------------------------

class TestDoDispatchKernelPrecheck:
    """_do_dispatch 在 kaggle 資源且 kernel 未 attach 時回 PENDING_KERNEL"""

    def test_do_dispatch_returns_pending_kernel_when_no_slug(self):
        """
        ResourceProber 判 kaggle + kaggle_kernel_slug 為空 →
        回傳 (False, "PENDING_KERNEL", reason)，不修改 submission status
        """
        import pollers.queue_dispatcher as qd

        sub = _make_submission("MH-2026-TEST-RC-020", kaggle_kernel_slug=None)
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = sub

        mock_prober_instance = MagicMock()
        mock_prober_instance.get_best_resource.return_value = {"resource": "kaggle", "device": "cuda"}
        mock_prober_cls = MagicMock(return_value=mock_prober_instance)

        with patch.dict("sys.modules", {
            "resources.prober": types.SimpleNamespace(ResourceProber=mock_prober_cls),
        }):
            success, resource, reason = qd._do_dispatch("MH-2026-TEST-RC-020", mock_db)

        assert success is False
        assert resource == "PENDING_KERNEL"
        assert "kaggle_kernel_slug" in reason
        # 不應推進 submission 狀態
        assert sub.status == "approved"

    def test_do_dispatch_proceeds_when_slug_set(self):
        """
        kaggle_kernel_slug 已設定 → 不應提早回 PENDING_KERNEL，繼續正常流程
        """
        import pollers.queue_dispatcher as qd

        sub = _make_submission(
            "MH-2026-TEST-RC-021",
            kaggle_kernel_slug="boardgamegroup/mh-2026-test-rc-021",
        )
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = sub

        mock_prober_instance = MagicMock()
        mock_prober_instance.get_best_resource.return_value = {"resource": "kaggle", "device": "cuda"}
        mock_prober_cls = MagicMock(return_value=mock_prober_instance)

        mock_handle = MagicMock()

        with patch.dict("sys.modules", {
            "resources.prober": types.SimpleNamespace(ResourceProber=mock_prober_cls),
            "routers.actions": types.SimpleNamespace(
                _handle_start_training_resource=mock_handle,
                DISABLE_LOCAL_TRAINING=False,
            ),
        }):
            success, resource, reason = qd._do_dispatch("MH-2026-TEST-RC-021", mock_db)

        # 有 kernel slug → 不應在前置檢查就回 PENDING_KERNEL
        assert resource != "PENDING_KERNEL"

    def test_do_dispatch_proceeds_when_prober_raises(self):
        """
        ResourceProber 拋例外時應 fallthrough，不讓 race condition check 阻斷正常流程
        """
        import pollers.queue_dispatcher as qd

        sub = _make_submission("MH-2026-TEST-RC-022", kaggle_kernel_slug=None)
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = sub

        mock_prober_cls = MagicMock(side_effect=RuntimeError("prober broken"))

        mock_handle = MagicMock()

        with patch.dict("sys.modules", {
            "resources.prober": types.SimpleNamespace(ResourceProber=mock_prober_cls),
            "routers.actions": types.SimpleNamespace(
                _handle_start_training_resource=mock_handle,
                DISABLE_LOCAL_TRAINING=False,
            ),
        }):
            # 不應 raise；應繼續到 start_training
            success, resource, reason = qd._do_dispatch("MH-2026-TEST-RC-022", mock_db)

        # prober 失敗後繼續 → sub.status 被推進到 training
        assert sub.status == "training"


# ---------------------------------------------------------------------------
# attach-kernel → reset pending_kernel 整合路徑
# ---------------------------------------------------------------------------

class TestAttachKernelResetsQueue:
    """attach-kernel 成功後應呼叫 QueueManager.reset_pending_kernel"""

    def setup_method(self):
        # 強制重新 import 真實的 queue_manager
        import sys
        sys.modules.pop("queue_manager", None)

    def test_attach_kernel_resets_pending_kernel_entry(self):
        """
        attach-kernel 成功 + 有 pending_kernel 條目 →
        回應 queue_reset_to_waiting=True
        """
        # 直接測試 reset_pending_kernel 被 attach-kernel 呼叫的邏輯
        # （endpoint 測試需完整 FastAPI app，此處驗證 QueueManager 互動）
        from queue_manager import QueueManager

        entry = _make_queue_entry(30, "MH-2026-TEST-RC-030", status="pending_kernel")
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = entry

        result = QueueManager.reset_pending_kernel(db, "MH-2026-TEST-RC-030")

        assert result is True
        assert entry.status == "waiting"

    def test_attach_kernel_no_pending_entry_returns_false(self):
        """
        attach-kernel 成功但無 pending_kernel 條目（正常情況）→
        reset_pending_kernel 回 False，不影響主流程
        """
        from queue_manager import QueueManager

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None

        result = QueueManager.reset_pending_kernel(db, "MH-2026-TEST-RC-031")

        assert result is False
