"""
tests/test_kaggle_force_rerun.py — 驗證 KaggleLauncher dispatch timestamp 注入邏輯

修復 bug：MH-019 v2 / MH-028 v3 push 沒觸發新訓練（Kaggle cache hit）。
根因：push 同一 kernel 時 code_file 內容相同 → Kaggle 不建新 version → 回傳舊 complete 狀態。
修法：push 前注入 MH_DISPATCH_TS comment → Kaggle 強制建新 version。

Test cases:
1. _inject_dispatch_ts: 全新 script 插入 marker 到第一行
2. _inject_dispatch_ts: 更新已有舊 marker（idempotent）
3. _remove_dispatch_ts: 移除 marker 恢復原始 script
4. _remove_dispatch_ts: 無 marker 時不修改（no-op）
5. push_and_attach: push 成功時 script 恢復 clean（no ts marker 殘留）
6. push_and_attach: push 失敗（returncode!=0）時 script 同樣恢復 clean
7. push_and_attach: force_rerun=True 時仍注入 timestamp（明確 override）
8. push_and_attach: 注入 ts 後 code 確實與注入前不同（確認 Kaggle 會建新 version）
9. get_current_kernel_version: 正確解析 "Version 5" 格式
10. get_current_kernel_version: 找不到 version 時回傳 None
"""

import json
import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# mock 掉 models 和 notifications，不需要真實 DB
import types as _types

if "models" not in sys.modules:
    _m = MagicMock()
    sys.modules["models"] = _m
if "notifications" not in sys.modules:
    _n = _types.ModuleType("notifications")
    _n.notify = MagicMock()
    _n.notify_event = MagicMock()
    sys.modules["notifications"] = _n

from resources.kaggle_launcher import (
    KaggleLauncher,
    _TS_MARKER,
    _inject_dispatch_ts,
    _remove_dispatch_ts,
    get_current_kernel_version,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kernel_dir(tmp_path: Path, code_content: str = "print('train')\n") -> Path:
    """建立最小化 kernel dir，含 kernel-metadata.json + train_kaggle.py"""
    kdir = tmp_path / "test_kernel"
    kdir.mkdir()
    meta = {
        "id": "testuser/test-kernel",
        "title": "Test Kernel",
        "code_file": "train_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "req_no": "MH-2026-TEST",
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta))
    (kdir / "train_kaggle.py").write_text(code_content)
    return kdir


def _make_submission_mock(req_no: str = "MH-2026-TEST") -> MagicMock:
    sub = MagicMock()
    sub.req_no = req_no
    sub.kaggle_kernel_slug = None
    sub.kaggle_status = None
    return sub


# ---------------------------------------------------------------------------
# Case 1: _inject_dispatch_ts — 全新 script 插入 marker 到第一行
# ---------------------------------------------------------------------------

class TestInjectDispatchTs:

    def test_inject_into_fresh_script(self, tmp_path):
        """全新 script（無 marker）→ marker 插到第一行"""
        script = tmp_path / "train.py"
        script.write_text("print('hello')\n")

        _inject_dispatch_ts(script, "20260427T120000Z")

        lines = script.read_text().splitlines()
        assert lines[0] == f"{_TS_MARKER} 20260427T120000Z"
        assert lines[1] == "print('hello')"

    # ---------------------------------------------------------------------------
    # Case 2: _inject_dispatch_ts — 更新舊 marker（idempotent）
    # ---------------------------------------------------------------------------

    def test_update_existing_marker(self, tmp_path):
        """已有舊 marker → 只更新時間戳，不重複插入"""
        script = tmp_path / "train.py"
        script.write_text(f"{_TS_MARKER} 20260426T000000Z\nprint('hello')\n")

        _inject_dispatch_ts(script, "20260427T120000Z")

        lines = script.read_text().splitlines()
        # 只有一行 marker
        marker_lines = [l for l in lines if l.startswith(_TS_MARKER)]
        assert len(marker_lines) == 1
        assert "20260427T120000Z" in marker_lines[0]
        assert "20260426T000000Z" not in script.read_text()


# ---------------------------------------------------------------------------
# Case 3: _remove_dispatch_ts — 移除 marker 恢復原始 script
# ---------------------------------------------------------------------------

class TestRemoveDispatchTs:

    def test_remove_existing_marker(self, tmp_path):
        """有 marker 時移除，還原 script"""
        original = "print('train')\n"
        script = tmp_path / "train.py"
        script.write_text(f"{_TS_MARKER} 20260427T120000Z\n{original}")

        _remove_dispatch_ts(script)

        assert script.read_text() == original

    # ---------------------------------------------------------------------------
    # Case 4: _remove_dispatch_ts — 無 marker 時不修改（no-op）
    # ---------------------------------------------------------------------------

    def test_no_op_when_no_marker(self, tmp_path):
        """無 marker → 不修改"""
        original = "print('hello')\n"
        script = tmp_path / "train.py"
        script.write_text(original)

        _remove_dispatch_ts(script)

        assert script.read_text() == original


# ---------------------------------------------------------------------------
# Case 5: push_and_attach — push 成功時 script 恢復 clean
# ---------------------------------------------------------------------------

class TestPushAndAttachCleanup:

    def test_script_clean_after_successful_push(self, tmp_path):
        """push 成功 → script 不留 ts marker"""
        kdir = _make_kernel_dir(tmp_path, "print('train')\n")
        db = MagicMock()
        sub = _make_submission_mock()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Kernel version 6 successfully pushed."
        mock_result.stderr = ""

        with patch("resources.kaggle_launcher.shutil.which", return_value="/usr/bin/kaggle"), \
             patch("resources.kaggle_launcher.get_kernel_dir", return_value=kdir), \
             patch("resources.kaggle_launcher.subprocess.run", return_value=mock_result), \
             patch("resources.kaggle_launcher.get_current_kernel_version", return_value=5):

            launcher = KaggleLauncher()
            result = launcher.push_and_attach("MH-2026-TEST", db, sub)

        assert result is True
        # script 不能含 marker
        script_content = (kdir / "train_kaggle.py").read_text()
        assert _TS_MARKER not in script_content

    # ---------------------------------------------------------------------------
    # Case 6: push_and_attach — push 失敗時 script 同樣恢復 clean
    # ---------------------------------------------------------------------------

    def test_script_clean_after_failed_push(self, tmp_path):
        """push 失敗（returncode=1）→ script 依然不留 ts marker"""
        kdir = _make_kernel_dir(tmp_path, "print('train')\n")
        db = MagicMock()
        sub = _make_submission_mock()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "404 Not Found"

        with patch("resources.kaggle_launcher.shutil.which", return_value="/usr/bin/kaggle"), \
             patch("resources.kaggle_launcher.get_kernel_dir", return_value=kdir), \
             patch("resources.kaggle_launcher.subprocess.run", return_value=mock_result), \
             patch("resources.kaggle_launcher.get_current_kernel_version", return_value=None):

            launcher = KaggleLauncher()
            result = launcher.push_and_attach("MH-2026-TEST", db, sub)

        assert result is False
        script_content = (kdir / "train_kaggle.py").read_text()
        assert _TS_MARKER not in script_content


# ---------------------------------------------------------------------------
# Case 7: push_and_attach — force_rerun=True 時仍注入 timestamp
# ---------------------------------------------------------------------------

class TestForceRerun:

    def test_force_rerun_still_injects_ts(self, tmp_path):
        """force_rerun=True → 明確 override，ts 注入後清除，push 成功"""
        kdir = _make_kernel_dir(tmp_path)
        db = MagicMock()
        sub = _make_submission_mock()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Kernel version 7 successfully pushed."
        mock_result.stderr = ""

        calls_with_marker = []

        original_run = subprocess.run if hasattr(mock, "_original_run") else None

        def capture_run(cmd, **kwargs):
            # 在 subprocess.run 執行時（即 push 時）捕捉 script 狀態
            script = kdir / "train_kaggle.py"
            calls_with_marker.append(_TS_MARKER in script.read_text())
            return mock_result

        with patch("resources.kaggle_launcher.shutil.which", return_value="/usr/bin/kaggle"), \
             patch("resources.kaggle_launcher.get_kernel_dir", return_value=kdir), \
             patch("resources.kaggle_launcher.subprocess.run", side_effect=capture_run), \
             patch("resources.kaggle_launcher.get_current_kernel_version", return_value=6):

            launcher = KaggleLauncher()
            result = launcher.push_and_attach("MH-2026-TEST", db, sub, force_rerun=True)

        assert result is True
        # push 時 script 應含 marker
        assert any(calls_with_marker), "ts marker should be present during push"
        # push 後 script 應乾淨
        assert _TS_MARKER not in (kdir / "train_kaggle.py").read_text()


# ---------------------------------------------------------------------------
# Case 8: 注入 ts 後 code 確實與注入前不同
# ---------------------------------------------------------------------------

class TestTsInjectionMakesCodeDifferent:

    def test_injected_code_differs_from_original(self, tmp_path):
        """注入 ts 後 code content 必定不同（確認 Kaggle 會建新 version）"""
        script = tmp_path / "train.py"
        original = "print('train')\n"
        script.write_text(original)

        original_content = script.read_text()
        _inject_dispatch_ts(script, "20260427T120000Z")
        injected_content = script.read_text()

        assert injected_content != original_content
        assert _TS_MARKER in injected_content

        # 不同時間戳注入後也不同
        _inject_dispatch_ts(script, "20260427T130000Z")
        injected_v2 = script.read_text()
        assert "20260427T130000Z" in injected_v2
        assert "20260427T120000Z" not in injected_v2


# ---------------------------------------------------------------------------
# Case 9: get_current_kernel_version — 正確解析 "Version 5"
# ---------------------------------------------------------------------------

class TestGetCurrentKernelVersion:

    def test_parse_version_number(self):
        """正確解析 kaggle status 輸出中的 Version N"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'boardgamegroup/mh-2026-019-site-object-detection-yolov8s has status "complete" - Version 5'

        with patch("resources.kaggle_launcher.shutil.which", return_value="/usr/bin/kaggle"), \
             patch("resources.kaggle_launcher.subprocess.run", return_value=mock_result):

            version = get_current_kernel_version("boardgamegroup/test-kernel")

        assert version == 5

    # ---------------------------------------------------------------------------
    # Case 10: get_current_kernel_version — 找不到 version 時回傳 None
    # ---------------------------------------------------------------------------

    def test_returns_none_when_version_not_found(self):
        """status 輸出無 Version 欄位時回傳 None"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'boardgamegroup/test has status "queued"'

        with patch("resources.kaggle_launcher.shutil.which", return_value="/usr/bin/kaggle"), \
             patch("resources.kaggle_launcher.subprocess.run", return_value=mock_result):

            version = get_current_kernel_version("boardgamegroup/test")

        assert version is None

    def test_returns_none_when_kaggle_not_found(self):
        """kaggle CLI 不存在時回傳 None"""
        with patch("resources.kaggle_launcher.shutil.which", return_value=None):
            version = get_current_kernel_version("boardgamegroup/test")
        assert version is None


import subprocess
