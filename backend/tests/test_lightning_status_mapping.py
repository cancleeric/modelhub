"""
Sprint 22 Task 22-1：Lightning status 正規化映射測試

確保 _normalize_status() 涵蓋所有已知 Status enum 值，
並對未知狀態 passthrough（lowercase）。
"""

import pytest
import sys
import os

# 確保 backend 目錄在 path 上
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pollers.lightning_poller import _normalize_status


class TestNormalizeStatus:
    """_normalize_status() 全覆蓋測試"""

    # --- running 狀態 ---
    def test_running_maps_to_running(self):
        assert _normalize_status("running") == "running"

    def test_running_case_insensitive(self):
        assert _normalize_status("Running") == "running"
        assert _normalize_status("RUNNING") == "running"

    # --- complete 狀態 ---
    def test_stopped_maps_to_complete(self):
        """Lightning Studio.stop() 後狀態為 stopped → modelhub complete"""
        assert _normalize_status("stopped") == "complete"

    # --- queued / pending 狀態 ---
    def test_queued_maps_to_queued(self):
        assert _normalize_status("queued") == "queued"

    def test_pending_maps_to_queued(self):
        assert _normalize_status("pending") == "queued"

    def test_starting_maps_to_queued(self):
        """Studio 啟動中"""
        assert _normalize_status("starting") == "queued"

    # --- error 狀態 ---
    def test_failed_maps_to_error(self):
        assert _normalize_status("failed") == "error"

    def test_error_maps_to_error(self):
        assert _normalize_status("error") == "error"

    # --- 未知狀態 passthrough ---
    def test_unknown_status_passthrough(self):
        """未知狀態應 lowercase passthrough，不 raise"""
        result = _normalize_status("SomeNewStatus")
        assert result == "somenewstatus"

    def test_empty_string_passthrough(self):
        result = _normalize_status("")
        assert result == ""

    # --- 常見 Lightning SDK 狀態完整確認 ---
    @pytest.mark.parametrize("raw,expected", [
        ("running",  "running"),
        ("stopped",  "complete"),
        ("failed",   "error"),
        ("error",    "error"),
        ("queued",   "queued"),
        ("pending",  "queued"),
        ("starting", "queued"),
        ("Stopping", "stopping"),   # Stopping 尚未映射 → passthrough
        ("NotCreated", "notcreated"),  # NotCreated → passthrough
    ])
    def test_parametrized_mapping(self, raw, expected):
        assert _normalize_status(raw) == expected
