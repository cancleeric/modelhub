"""
test_kaggle_poller_metrics.py — 防止 kaggle_poller metrics 解析 regression

覆蓋場景：
1. _read_result_json：YOLO 格式（map50/map50_95）正確解析
2. _read_result_json：OCR 格式（cer/exact_match）正確對應 map50
3. _read_result_json：無 result.json → 回傳 {}（不 crash）
4. _read_result_json：partial OCR（只有 val_cer）
5. _on_kernel_complete：per_class_metrics fallback（map50=None 時從 sub 算均值）
6. _compute_pass_fail：邊界值
7. _on_kernel_complete：OCR 任務 notes 含 CER

修復記錄：2026-04-27 MH-006/MH-009 回填，root cause = _read_result_json 缺 OCR 支援
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@contextmanager
def _tmp_dir():
    """
    容器內 /tmp 磁碟可能滿（overlay fs）；改用 /app/tests-tmp/（bind mount host）
    或 host 上的系統 tmp（fallback），自動清理。
    """
    # 優先在 /app 下（容器內 bind-mounted host 路徑，空間充足）
    base_candidates = [
        Path("/app/tests-tmp"),
        Path(tempfile.gettempdir()),
    ]
    base = next((b for b in base_candidates if _check_writable(b)), Path(tempfile.gettempdir()))
    base.mkdir(parents=True, exist_ok=True)
    d = tempfile.mkdtemp(dir=str(base))
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _check_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.touch()
        probe.unlink()
        return True
    except Exception:
        return False


def _write_result_json(tmp_dir: Path, data: dict) -> str:
    (tmp_dir / "result.json").write_text(json.dumps(data))
    return str(tmp_dir)


def _make_sub(req_no: str, arch: str = "yolov8m",
              per_class_metrics=None, map50_threshold: float = 0.5,
              status: str = "training", kaggle_status: str = "running"):
    sub = MagicMock()
    sub.req_no = req_no
    sub.status = status
    sub.kaggle_status = kaggle_status
    sub.kaggle_kernel_slug = f"boardgamegroup/{req_no.lower()}"
    sub.arch = arch
    sub.product = "TestProduct"
    sub.req_name = req_no
    sub.training_started_at = None
    sub.retry_count = 0
    sub.max_retries = 2
    sub.budget_exceeded_notified = False
    sub.max_budget_usd = 0
    sub.per_class_metrics = (
        json.dumps(per_class_metrics) if isinstance(per_class_metrics, dict)
        else per_class_metrics
    )
    sub.gpu_seconds = None
    sub.estimated_cost_usd = None
    sub.training_completed_at = None
    sub.kaggle_status_updated_at = None
    sub.map50_threshold = map50_threshold
    return sub


# ---------------------------------------------------------------------------
# 1. _read_result_json — YOLO 格式
# ---------------------------------------------------------------------------

class TestReadResultJsonYolo:

    def test_yolo_map50_map50_95(self):
        """YOLO result.json 含 map50 / map50_95 應正確解析"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with _tmp_dir() as d:
            dest = _write_result_json(d, {
                "map50": 0.895,
                "map50_95": 0.623,
                "epochs": 100,
                "batch_size": 16,
            })
            result = kp._read_result_json(dest)

        assert result["metrics"]["map50"] == pytest.approx(0.895)
        assert result["metrics"]["map50_95"] == pytest.approx(0.623)
        assert result["metrics"]["epochs"] == 100
        assert result["metrics"]["batch_size"] == 16
        assert result["per_class"] is None

    def test_yolo_per_class_map50(self):
        """含 per_class_map50 應回傳 per_class dict"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with _tmp_dir() as d:
            dest = _write_result_json(d, {
                "map50": 0.750,
                "per_class_map50": {"helmet": 0.82, "vest": 0.68},
            })
            result = kp._read_result_json(dest)

        assert result["metrics"]["map50"] == pytest.approx(0.750)
        assert result["per_class"] == {"helmet": 0.82, "vest": 0.68}

    def test_yolo_only_map50_no_map50_95(self):
        """只有 map50，無 map50_95 → map50_95 不在 metrics"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with _tmp_dir() as d:
            dest = _write_result_json(d, {"map50": 0.4489})
            result = kp._read_result_json(dest)

        assert result["metrics"]["map50"] == pytest.approx(0.4489)
        assert "map50_95" not in result["metrics"]


# ---------------------------------------------------------------------------
# 2. _read_result_json — OCR 格式（Bug fix regression test）
# ---------------------------------------------------------------------------

class TestReadResultJsonOcr:

    def test_ocr_test_exact_match_becomes_map50(self):
        """
        OCR result.json：test_exact_match → map50，test_cer → ocr_cer。
        這是 MH-009 的 fix regression test。
        """
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with _tmp_dir() as d:
            dest = _write_result_json(d, {
                "test_exact_match": 0.036,
                "test_cer": 0.4227,
                "val_cer": 0.4152,
                "epochs": 8,
            })
            result = kp._read_result_json(dest)

        # exact_match 應映射到 map50
        assert result["metrics"]["map50"] == pytest.approx(0.036)
        # CER 應保存（test_cer 優先於 val_cer）
        assert result["metrics"]["ocr_cer"] == pytest.approx(0.4227)
        # epochs 應解析
        assert result["metrics"]["epochs"] == 8

    def test_ocr_exact_match_fallback(self):
        """exact_match（無 test_ prefix）也應被識別"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with _tmp_dir() as d:
            dest = _write_result_json(d, {
                "exact_match": 0.15,
                "cer": 0.30,
            })
            result = kp._read_result_json(dest)

        assert result["metrics"]["map50"] == pytest.approx(0.15)
        assert result["metrics"]["ocr_cer"] == pytest.approx(0.30)

    def test_ocr_only_cer_no_exact_match(self):
        """只有 val_cer，無 exact_match → ocr_cer 存在，但 map50 不應是 cer 值"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with _tmp_dir() as d:
            dest = _write_result_json(d, {"val_cer": 0.42})
            result = kp._read_result_json(dest)

        # 沒有 exact_match → map50 不應被設（避免把 CER 錯誤當 map50）
        assert "map50" not in result["metrics"]
        assert result["metrics"]["ocr_cer"] == pytest.approx(0.42)

    def test_yolo_map50_takes_precedence_over_exact_match(self):
        """
        若同時有 map50 和 exact_match，應以 map50 優先
        （不應發生，但防衛性測試）
        """
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with _tmp_dir() as d:
            dest = _write_result_json(d, {
                "map50": 0.85,
                "exact_match": 0.50,
            })
            result = kp._read_result_json(dest)

        # map50 已存在，exact_match 不覆蓋
        assert result["metrics"]["map50"] == pytest.approx(0.85)

    def test_no_result_json_returns_empty(self):
        """不存在 result.json → 回傳 {}"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with _tmp_dir() as d:
            result = kp._read_result_json(str(d))
        assert result == {}

    def test_malformed_json_returns_empty(self):
        """result.json 內容非 JSON → 回傳 {}，不 crash"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        with _tmp_dir() as d:
            (d / "result.json").write_text("not valid json {{{")
            result = kp._read_result_json(str(d))
        assert result == {}


# ---------------------------------------------------------------------------
# 3. per_class_metrics fallback（Bug fix regression test）
# ---------------------------------------------------------------------------

class TestPerClassMetricsFallback:

    @pytest.mark.asyncio
    async def test_per_class_fallback_when_no_map50_in_output(self):
        """
        Kaggle kernel output 沒有 result.json（或無 map50），
        sub.per_class_metrics = {"text": 0.4489} → map50 = 0.4489。
        這是 MH-006 v3 的 fix regression test。
        """
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        sub = _make_sub(
            "MH-2026-006",
            arch="yolov8s",
            per_class_metrics={"text": 0.4489},
            map50_threshold=0.5974,
        )

        created_mv = {}

        def fake_mv_cls(**kwargs):
            created_mv.update(kwargs)
            return MagicMock(**kwargs)

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = MagicMock(version="v2")

        with _tmp_dir() as d:
            with patch.object(kp, "_download_kernel_output", new=AsyncMock(return_value=str(d))), \
                 patch.object(kp, "_read_result_json", return_value={}), \
                 patch.object(kp, "_read_log_files", return_value=""), \
                 patch.object(kp, "parse_training_log", return_value={"metrics": {}}), \
                 patch.object(kp, "_next_version_for", return_value="v3"), \
                 patch.object(kp, "_append_history"), \
                 patch.object(kp, "notify_event", new=AsyncMock()), \
                 patch.object(kp, "ModelVersion", side_effect=fake_mv_cls), \
                 patch("pollers.kaggle_poller.SessionLocal") as mock_sl:

                mock_sl.return_value = MagicMock()
                await kp._on_kernel_complete(db, sub)

        # per_class fallback 應算出 map50=0.4489
        assert "map50" in created_mv
        assert created_mv["map50"] == pytest.approx(0.4489)
        # pass_fail 應為 fail（0.4489 < 0.5974）
        assert created_mv["pass_fail"] == "fail"

    @pytest.mark.asyncio
    async def test_per_class_multi_class_average(self):
        """多 class 的 per_class_metrics → map50 應為均值"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        sub = _make_sub(
            "MH-TEST-001",
            per_class_metrics={"helmet": 0.80, "vest": 0.60, "glove": 0.70},
            map50_threshold=0.65,
        )

        created_mv = {}

        def fake_mv_cls(**kwargs):
            created_mv.update(kwargs)
            return MagicMock(**kwargs)

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = MagicMock(version="v0")

        with _tmp_dir() as d:
            with patch.object(kp, "_download_kernel_output", new=AsyncMock(return_value=str(d))), \
                 patch.object(kp, "_read_result_json", return_value={}), \
                 patch.object(kp, "_read_log_files", return_value=""), \
                 patch.object(kp, "parse_training_log", return_value={"metrics": {}}), \
                 patch.object(kp, "_next_version_for", return_value="v1"), \
                 patch.object(kp, "_append_history"), \
                 patch.object(kp, "notify_event", new=AsyncMock()), \
                 patch.object(kp, "ModelVersion", side_effect=fake_mv_cls), \
                 patch("pollers.kaggle_poller.SessionLocal") as mock_sl:

                mock_sl.return_value = MagicMock()
                await kp._on_kernel_complete(db, sub)

        # 均值 = (0.80+0.60+0.70)/3 = 0.70
        assert created_mv["map50"] == pytest.approx(0.70, abs=1e-4)
        # 0.70 >= 0.65 → pass
        assert created_mv["pass_fail"] == "pass"

    @pytest.mark.asyncio
    async def test_no_fallback_when_map50_present_in_result(self):
        """result.json 已有 map50 → 不觸發 per_class fallback"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        sub = _make_sub(
            "MH-TEST-002",
            per_class_metrics={"text": 0.9999},  # 若 fallback 觸發，map50 會是 0.9999
            map50_threshold=0.5,
        )

        created_mv = {}

        def fake_mv_cls(**kwargs):
            created_mv.update(kwargs)
            return MagicMock(**kwargs)

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = MagicMock(version="v0")

        with _tmp_dir() as d:
            # result.json 已有 map50=0.65 → 應使用這個值
            with patch.object(kp, "_download_kernel_output", new=AsyncMock(return_value=str(d))), \
                 patch.object(kp, "_read_result_json",
                              return_value={"metrics": {"map50": 0.65}, "per_class": None}), \
                 patch.object(kp, "_next_version_for", return_value="v1"), \
                 patch.object(kp, "_append_history"), \
                 patch.object(kp, "notify_event", new=AsyncMock()), \
                 patch.object(kp, "ModelVersion", side_effect=fake_mv_cls), \
                 patch("pollers.kaggle_poller.SessionLocal") as mock_sl:

                mock_sl.return_value = MagicMock()
                await kp._on_kernel_complete(db, sub)

        # 應使用 result.json 的 0.65，不用 per_class 的 0.9999
        assert created_mv["map50"] == pytest.approx(0.65)

    @pytest.mark.asyncio
    async def test_no_fallback_when_per_class_is_none(self):
        """sub.per_class_metrics=None 且 result.json 無 map50 → map50=None"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        sub = _make_sub("MH-TEST-003", per_class_metrics=None, map50_threshold=0.5)

        created_mv = {}

        def fake_mv_cls(**kwargs):
            created_mv.update(kwargs)
            return MagicMock(**kwargs)

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = MagicMock(version="v0")

        with _tmp_dir() as d:
            with patch.object(kp, "_download_kernel_output", new=AsyncMock(return_value=str(d))), \
                 patch.object(kp, "_read_result_json", return_value={}), \
                 patch.object(kp, "_read_log_files", return_value=""), \
                 patch.object(kp, "parse_training_log", return_value={"metrics": {}}), \
                 patch.object(kp, "_next_version_for", return_value="v1"), \
                 patch.object(kp, "_append_history"), \
                 patch.object(kp, "notify_event", new=AsyncMock()), \
                 patch.object(kp, "ModelVersion", side_effect=fake_mv_cls), \
                 patch("pollers.kaggle_poller.SessionLocal") as mock_sl:

                mock_sl.return_value = MagicMock()
                await kp._on_kernel_complete(db, sub)

        assert created_mv["map50"] is None
        assert created_mv["pass_fail"] is None


# ---------------------------------------------------------------------------
# 4. _compute_pass_fail 邊界值
# ---------------------------------------------------------------------------

class TestComputePassFail:

    def test_pass_at_threshold(self):
        import pollers.kaggle_poller as kp
        assert kp._compute_pass_fail(0.5, 0.5) == "pass"

    def test_fail_below_threshold(self):
        import pollers.kaggle_poller as kp
        assert kp._compute_pass_fail(0.4999, 0.5) == "fail"

    def test_pass_above_threshold(self):
        import pollers.kaggle_poller as kp
        assert kp._compute_pass_fail(0.895, 0.7) == "pass"

    def test_fail_exact_mh006(self):
        """MH-006 v3: map50=0.4489 < threshold=0.5974 → fail"""
        import pollers.kaggle_poller as kp
        assert kp._compute_pass_fail(0.4489, 0.5974) == "fail"

    def test_pass_exact_mh018(self):
        """MH-018 v1: map50=0.895 >= threshold=0.70 → pass"""
        import pollers.kaggle_poller as kp
        assert kp._compute_pass_fail(0.895, 0.70) == "pass"


# ---------------------------------------------------------------------------
# 5. OCR 任務 ModelVersion notes 含 CER
# ---------------------------------------------------------------------------

class TestOcrNotesContainCer:

    @pytest.mark.asyncio
    async def test_ocr_cer_in_notes(self):
        """OCR 任務 result.json 有 ocr_cer → ModelVersion.notes 應含 CER 值"""
        import importlib
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        sub = _make_sub(
            "MH-2026-009",
            arch="TrOCR-small",
            per_class_metrics=None,
            map50_threshold=0.5,
        )

        created_mv = {}

        def fake_mv_cls(**kwargs):
            created_mv.update(kwargs)
            return MagicMock(**kwargs)

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = MagicMock(version="v0")

        with _tmp_dir() as d:
            # _read_result_json 回傳 OCR 格式（已修 fix）
            with patch.object(kp, "_download_kernel_output", new=AsyncMock(return_value=str(d))), \
                 patch.object(kp, "_read_result_json", return_value={
                     "metrics": {"map50": 0.036, "ocr_cer": 0.4227},
                     "per_class": None,
                 }), \
                 patch.object(kp, "_next_version_for", return_value="v1"), \
                 patch.object(kp, "_append_history"), \
                 patch.object(kp, "notify_event", new=AsyncMock()), \
                 patch.object(kp, "ModelVersion", side_effect=fake_mv_cls), \
                 patch("pollers.kaggle_poller.SessionLocal") as mock_sl:

                mock_sl.return_value = MagicMock()
                await kp._on_kernel_complete(db, sub)

        assert "CER=0.4227" in created_mv.get("notes", "")
        assert created_mv["map50"] == pytest.approx(0.036)
