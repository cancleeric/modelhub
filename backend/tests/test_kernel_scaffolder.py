"""
tests/test_kernel_scaffolder.py — M17 KernelScaffolder 單元測試

驗收條件：
1. scaffold 後 kaggle-kernels/ 目錄自動生成，slug 格式符合
   boardgamegroup/mh-YYYY-NNN-<name>
2. 生成的 train_kaggle.py 含正確 epochs、arch、class_count
3. 重複 scaffold 同一 req_no 不覆寫（returns existing dir）
4. scaffold_if_missing helper 行為正確
"""
import json
import sys
import types
import unittest.mock as mock

# ── 安裝最小 mock，避免 import 側效（DB / notifications / parsers）──────────
if "models" not in sys.modules:
    models_mock = types.ModuleType("models")
    models_mock.SessionLocal = mock.MagicMock()
    models_mock.Submission = mock.MagicMock()
    models_mock.ModelVersion = mock.MagicMock()
    models_mock.SubmissionHistory = mock.MagicMock()
    models_mock.TrainingQueue = mock.MagicMock()
    sys.modules["models"] = models_mock

if "notifications" not in sys.modules:
    notif_mock = types.ModuleType("notifications")
    notif_mock.notify = mock.AsyncMock(return_value=True)
    notif_mock.notify_event = mock.AsyncMock(return_value=None)
    notif_mock.CTO_TARGET = "cto@hurricanecore.internal"
    sys.modules["notifications"] = notif_mock

if "parsers" not in sys.modules:
    parsers_mock = types.ModuleType("parsers")
    parsers_mock.parse_training_log = lambda arch, log_text: {"metrics": {}, "per_class": {}}
    sys.modules["parsers"] = parsers_mock
# ───────────────────────────────────────────────────────────────────────────

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class _FakeSubmission:
    """最小 Submission mock"""
    def __init__(self, req_no="MH-2026-099", req_name="Test Model",
                 arch="yolov8m", class_count=5,
                 dataset_path=None, kaggle_dataset_url=None,
                 kaggle_kernel_slug=None):
        self.req_no = req_no
        self.req_name = req_name
        self.arch = arch
        self.class_count = class_count
        self.dataset_path = dataset_path
        self.kaggle_dataset_url = kaggle_dataset_url
        self.kaggle_kernel_slug = kaggle_kernel_slug


@pytest.fixture()
def tmp_kernels_root(tmp_path):
    """臨時 kaggle-kernels 根目錄"""
    return tmp_path / "kaggle-kernels"


@pytest.fixture()
def scaffolder(tmp_kernels_root):
    """KernelScaffolder with patched root & template dir"""
    from resources.kernel_scaffolder import KernelScaffolder, _TEMPLATES_DIR
    s = KernelScaffolder()
    # patch _KAGGLE_KERNELS_ROOT
    with patch("resources.kernel_scaffolder._KAGGLE_KERNELS_ROOT", tmp_kernels_root):
        yield s


class TestSlugGeneration:
    def test_slug_format(self):
        from resources.kernel_scaffolder import _make_slug
        sub = _FakeSubmission(req_no="MH-2026-001", req_name="Site Object Detection")
        slug = _make_slug(sub)
        # boardgamegroup/mh-2026-001-<name>
        assert slug.startswith("boardgamegroup/mh-2026-001-")
        # 全小寫
        assert slug == slug.lower()

    def test_slug_no_req_name(self):
        from resources.kernel_scaffolder import _make_slug
        sub = _FakeSubmission(req_no="MH-2026-005", req_name=None)
        slug = _make_slug(sub)
        assert slug == "boardgamegroup/mh-2026-005"

    def test_slug_special_chars_stripped(self):
        from resources.kernel_scaffolder import _make_slug
        sub = _FakeSubmission(req_no="MH-2026-010", req_name="Model (V2)  !!")
        slug = _make_slug(sub)
        assert "boardgamegroup/mh-2026-010-" in slug
        assert "!" not in slug
        assert "(" not in slug


class TestScaffold:
    def test_scaffold_creates_directory(self, tmp_kernels_root):
        from resources.kernel_scaffolder import KernelScaffolder
        s = KernelScaffolder()
        sub = _FakeSubmission(req_no="MH-2026-099", req_name="Test Model",
                              arch="yolov8m", class_count=3)
        with patch("resources.kernel_scaffolder._KAGGLE_KERNELS_ROOT", tmp_kernels_root):
            kernel_dir = s.scaffold(sub)

        assert kernel_dir.exists()
        assert (kernel_dir / "kernel-metadata.json").exists()
        assert (kernel_dir / "train_kaggle.py").exists()

    def test_metadata_slug_format(self, tmp_kernels_root):
        from resources.kernel_scaffolder import KernelScaffolder
        s = KernelScaffolder()
        sub = _FakeSubmission(req_no="MH-2026-099", req_name="Test Model",
                              arch="yolov8m", class_count=3)
        with patch("resources.kernel_scaffolder._KAGGLE_KERNELS_ROOT", tmp_kernels_root):
            kernel_dir = s.scaffold(sub)

        meta = json.loads((kernel_dir / "kernel-metadata.json").read_text())
        assert meta["id"].startswith("boardgamegroup/mh-2026-099-")
        assert meta["req_no"] == "MH-2026-099"
        assert meta["enable_gpu"] is True
        assert meta["kernel_type"] == "script"

    def test_train_py_contains_arch_epochs_class_count(self, tmp_kernels_root):
        from resources.kernel_scaffolder import KernelScaffolder
        s = KernelScaffolder()
        sub = _FakeSubmission(req_no="MH-2026-099", req_name="Test Model",
                              arch="yolov8n", class_count=7)
        with patch("resources.kernel_scaffolder._KAGGLE_KERNELS_ROOT", tmp_kernels_root):
            kernel_dir = s.scaffold(sub)

        content = (kernel_dir / "train_kaggle.py").read_text()
        assert "yolov8n" in content       # arch
        assert "100" in content           # default epochs
        assert "7" in content             # class_count (NUM_CLASSES = 7)
        assert "MH-2026-099" in content   # req_no

    def test_scaffold_no_overwrite_existing(self, tmp_kernels_root):
        """第二次 scaffold 同一 req_no，不覆寫既有目錄"""
        from resources.kernel_scaffolder import KernelScaffolder
        s = KernelScaffolder()
        sub = _FakeSubmission(req_no="MH-2026-099", req_name="Test Model",
                              arch="yolov8m", class_count=3)
        with patch("resources.kernel_scaffolder._KAGGLE_KERNELS_ROOT", tmp_kernels_root):
            dir1 = s.scaffold(sub)
            # 修改 train_kaggle.py 內容（模擬手動改動）
            sentinel = "# SENTINEL"
            (dir1 / "train_kaggle.py").write_text(sentinel)
            # 再次 scaffold
            dir2 = s.scaffold(sub)
            # 目錄相同，內容不被覆寫
            assert dir1 == dir2
            assert (dir2 / "train_kaggle.py").read_text() == sentinel

    def test_scaffold_with_kaggle_dataset_url(self, tmp_kernels_root):
        """dataset_sources 從 kaggle_dataset_url 正確生成"""
        from resources.kernel_scaffolder import KernelScaffolder
        s = KernelScaffolder()
        sub = _FakeSubmission(
            req_no="MH-2026-099", req_name="Test",
            arch="yolov8m", class_count=3,
            kaggle_dataset_url="boardgamegroup/my-test-dataset",
        )
        with patch("resources.kernel_scaffolder._KAGGLE_KERNELS_ROOT", tmp_kernels_root):
            kernel_dir = s.scaffold(sub)

        meta = json.loads((kernel_dir / "kernel-metadata.json").read_text())
        assert "boardgamegroup/my-test-dataset" in meta["dataset_sources"]

    def test_scaffold_if_missing_no_exception_on_error(self, tmp_kernels_root):
        """scaffold_if_missing 失敗時回傳 None，不 raise"""
        from resources.kernel_scaffolder import KernelScaffolder
        s = KernelScaffolder()
        sub = _FakeSubmission(req_no="MH-2026-099", req_name="Test", arch="yolov8m", class_count=3)

        with patch("resources.kernel_scaffolder._KAGGLE_KERNELS_ROOT", tmp_kernels_root):
            with patch.object(s, "scaffold", side_effect=RuntimeError("boom")):
                result = s.scaffold_if_missing(sub)

        assert result is None

    def test_modelhub_report_hook_present(self, tmp_kernels_root):
        """生成的 train_kaggle.py 末尾呼叫 modelhub_report"""
        from resources.kernel_scaffolder import KernelScaffolder
        s = KernelScaffolder()
        sub = _FakeSubmission(req_no="MH-2026-099", req_name="Test", arch="yolov8m", class_count=3)
        with patch("resources.kernel_scaffolder._KAGGLE_KERNELS_ROOT", tmp_kernels_root):
            kernel_dir = s.scaffold(sub)

        content = (kernel_dir / "train_kaggle.py").read_text()
        assert "modelhub_report" in content
