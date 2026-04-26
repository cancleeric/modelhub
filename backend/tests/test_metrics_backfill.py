"""
test_metrics_backfill.py — 驗證 mAP metrics 回寫邏輯

修復工單：MH-2026-019 mAP None bug

Root cause：
  1. parse_yolo_log 未支援 Kaggle NDJSON log 格式 → regex 全部 miss
  2. _on_kernel_complete 未嘗試直讀 result.json → metrics={} → ModelVersion map50=None
  3. pass_fail 未計算 → ModelVersion pass_fail=None

修復：
  A. parsers/yolo.py：_decode_ndjson_log + result.json 優先解析
  B. pollers/kaggle_poller.py：_read_result_json + _compute_pass_fail + 優先讀 result.json
"""

import sys
import os
import json
import importlib
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# 指向 /app（由 conftest.py 的 sys.path 設定）
_APP_DIR = os.path.join(os.path.dirname(__file__), "..")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


# ---------------------------------------------------------------------------
# Helpers：繞過 conftest.py 的 parsers mock，直接從實際模組匯入
# ---------------------------------------------------------------------------

def _import_parsers_yolo():
    """
    conftest.py 把 parsers 整個 mock 掉，需移除後重新載入真實模組。
    用完後不還原（測試隔離由 pytest 保證）。
    """
    for key in list(sys.modules.keys()):
        if key == "parsers" or key.startswith("parsers."):
            del sys.modules[key]
    import parsers.yolo as yolo_mod
    return yolo_mod


def _make_sub(req_no="MH-2026-019", map50_threshold=0.6, status="training",
              kaggle_status="complete", arch="yolov8s"):
    sub = MagicMock()
    sub.req_no = req_no
    sub.status = status
    sub.kaggle_status = kaggle_status
    sub.kaggle_kernel_slug = f"boardgamegroup/{req_no.lower()}"
    sub.arch = arch
    sub.product = "site-detection"
    sub.req_name = req_no
    sub.map50_threshold = map50_threshold
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


SAMPLE_RESULT_JSON = {
    "req_no": "MH-2026-019",
    "run": "kaggle_v1_cuda",
    "arch": "yolov8s",
    "device": "cuda",
    "epochs": 150,
    "map50": 0.7455,
    "map50_95": 0.5137,
    "per_class_map50": {
        "furniture_chair": 0.245,
        "furniture_sofa": 0.995,
    },
    "verdict": "pass",
}

SAMPLE_NDJSON_LOG = (
    '[{"stream_name":"stdout","time":1.0,"data":"Epoch 1/10\\n"}\n'
    ',{"stream_name":"stdout","time":2.0,"data":"                   all         6        14      0.900      0.857      0.746      0.514\\n"}\n'
    ',{"stream_name":"stdout","time":3.0,"data":"mAP50=0.7455  mAP50-95=0.5137\\n"}\n'
    ']'
)


# ---------------------------------------------------------------------------
# 1. _decode_ndjson_log
# ---------------------------------------------------------------------------

class TestDecodeNdjsonLog:

    def test_decodes_kaggle_ndjson_format(self):
        yolo_mod = _import_parsers_yolo()
        result = yolo_mod._decode_ndjson_log(SAMPLE_NDJSON_LOG)
        assert "Epoch 1/10" in result
        assert "all" in result
        assert "0.746" in result

    def test_plain_log_returned_unchanged(self):
        yolo_mod = _import_parsers_yolo()
        plain = "Epoch 1/10\n    all      6     14    0.900    0.857    0.746    0.514\n"
        result = yolo_mod._decode_ndjson_log(plain)
        assert result == plain

    def test_empty_string(self):
        yolo_mod = _import_parsers_yolo()
        assert yolo_mod._decode_ndjson_log("") == ""

    def test_ndjson_includes_stderr(self):
        """stderr 行也應被納入"""
        yolo_mod = _import_parsers_yolo()
        ndjson = (
            '[{"stream_name":"stdout","time":1.0,"data":"Training\\n"}\n'
            ',{"stream_name":"stderr","time":2.0,"data":"Warning: something\\n"}\n'
            ']'
        )
        result = yolo_mod._decode_ndjson_log(ndjson)
        assert "Training" in result
        assert "Warning" in result


# ---------------------------------------------------------------------------
# 2. parse_yolo_log — result.json 優先路徑
# ---------------------------------------------------------------------------

class TestParseYoloLogResultJson:

    def test_extracts_metrics_from_result_json_in_text(self):
        """log_text 中含 result.json 內容時，應直接取 map50/map50_95"""
        yolo_mod = _import_parsers_yolo()
        log_text = json.dumps(SAMPLE_RESULT_JSON)
        result = yolo_mod.parse_yolo_log(log_text)
        assert result["metrics"]["map50"] == pytest.approx(0.7455)
        assert result["metrics"]["map50_95"] == pytest.approx(0.5137)
        assert result["metrics"]["epochs"] == 150
        assert result.get("_source") == "result_json"

    def test_per_class_extracted_from_result_json(self):
        yolo_mod = _import_parsers_yolo()
        log_text = json.dumps(SAMPLE_RESULT_JSON)
        result = yolo_mod.parse_yolo_log(log_text)
        assert result["per_class"] is not None
        assert "furniture_sofa" in result["per_class"]

    def test_ndjson_log_fallback_when_no_result_json(self):
        """沒有 result.json 時，應 fallback 到 NDJSON 解碼 + regex"""
        yolo_mod = _import_parsers_yolo()
        result = yolo_mod.parse_yolo_log(SAMPLE_NDJSON_LOG)
        assert result["metrics"].get("map50") is not None
        assert result["metrics"]["map50"] == pytest.approx(0.746, abs=0.01)

    def test_result_json_takes_priority_over_ndjson(self):
        """result.json 和 NDJSON 都在 text 時，result.json 優先"""
        yolo_mod = _import_parsers_yolo()
        combined = json.dumps(SAMPLE_RESULT_JSON) + "\n" + SAMPLE_NDJSON_LOG
        result = yolo_mod.parse_yolo_log(combined)
        assert result["metrics"]["map50"] == pytest.approx(0.7455)
        assert result.get("_source") == "result_json"


# ---------------------------------------------------------------------------
# 3. _read_result_json
# ---------------------------------------------------------------------------

class TestReadResultJson:

    def test_reads_metrics_from_result_json(self, tmp_path):
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        (tmp_path / "result.json").write_text(json.dumps(SAMPLE_RESULT_JSON))
        result = kp._read_result_json(str(tmp_path))
        assert result["metrics"]["map50"] == pytest.approx(0.7455)
        assert result["metrics"]["map50_95"] == pytest.approx(0.5137)
        assert result["per_class"]["furniture_sofa"] == pytest.approx(0.995)

    def test_returns_empty_when_no_file(self, tmp_path):
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        result = kp._read_result_json(str(tmp_path))
        assert result == {}

    def test_returns_empty_when_malformed_json(self, tmp_path):
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        (tmp_path / "result.json").write_text("not-valid-json{")
        result = kp._read_result_json(str(tmp_path))
        assert result == {}

    def test_returns_empty_metrics_when_missing_map50_key(self, tmp_path):
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        (tmp_path / "result.json").write_text(json.dumps({"req_no": "MH-2026-019"}))
        result = kp._read_result_json(str(tmp_path))
        assert result.get("metrics", {}).get("map50") is None


# ---------------------------------------------------------------------------
# 4. _compute_pass_fail
# ---------------------------------------------------------------------------

class TestComputePassFail:

    def test_pass_when_map50_above_threshold(self):
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        assert kp._compute_pass_fail(0.7455, 0.6) == "pass"

    def test_fail_when_map50_below_threshold(self):
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        assert kp._compute_pass_fail(0.45, 0.6) == "fail"

    def test_pass_when_exactly_at_threshold(self):
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        assert kp._compute_pass_fail(0.6, 0.6) == "pass"

    def test_pass_with_zero_threshold(self):
        """threshold=0 時任何正值 map50 都是 pass"""
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        assert kp._compute_pass_fail(0.01, 0.0) == "pass"


# ---------------------------------------------------------------------------
# 5. _on_kernel_complete — end-to-end metrics 回寫
#
# 注意：ModelVersion 被 conftest mock，db.add() 收到的 obj 是真實 ModelVersion
# instance 的模擬品。我們改用 patch('models.ModelVersion') 捕捉 kwargs。
# ---------------------------------------------------------------------------

class TestOnKernelCompleteMetrics:

    @pytest.mark.asyncio
    async def test_writes_map50_and_pass_fail_from_result_json(self, tmp_path):
        """
        _on_kernel_complete：result.json 存在時，ModelVersion 應有正確
        map50 / map50_95 / pass_fail 值。
        """
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        (tmp_path / "result.json").write_text(json.dumps(SAMPLE_RESULT_JSON))

        sub = _make_sub(map50_threshold=0.6)
        db = MagicMock()
        mv_kwargs: dict = {}

        # Capture the kwargs passed to ModelVersion(...)
        original_mv = kp.ModelVersion

        class CapturingMV:
            def __init__(self, **kwargs):
                mv_kwargs.update(kwargs)

        with patch.object(kp, "_download_kernel_output",
                          new=AsyncMock(return_value=str(tmp_path))), \
             patch.object(kp, "_read_log_files", return_value=""), \
             patch.object(kp, "_next_version_for", return_value="v1"), \
             patch.object(kp, "_append_history"), \
             patch.object(kp, "notify_event", new=AsyncMock()), \
             patch.object(kp, "ModelVersion", CapturingMV):

            await kp._on_kernel_complete(db, sub)

        assert mv_kwargs.get("map50") == pytest.approx(0.7455)
        assert mv_kwargs.get("map50_95") == pytest.approx(0.5137)
        assert mv_kwargs.get("pass_fail") == "pass"

    @pytest.mark.asyncio
    async def test_writes_fail_when_below_threshold(self, tmp_path):
        """map50 < threshold → pass_fail='fail'"""
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        low_result = dict(SAMPLE_RESULT_JSON)
        low_result["map50"] = 0.45
        (tmp_path / "result.json").write_text(json.dumps(low_result))

        sub = _make_sub(map50_threshold=0.6)
        db = MagicMock()
        mv_kwargs: dict = {}

        class CapturingMV:
            def __init__(self, **kwargs):
                mv_kwargs.update(kwargs)

        with patch.object(kp, "_download_kernel_output",
                          new=AsyncMock(return_value=str(tmp_path))), \
             patch.object(kp, "_read_log_files", return_value=""), \
             patch.object(kp, "_next_version_for", return_value="v1"), \
             patch.object(kp, "_append_history"), \
             patch.object(kp, "notify_event", new=AsyncMock()), \
             patch.object(kp, "ModelVersion", CapturingMV):

            await kp._on_kernel_complete(db, sub)

        assert mv_kwargs.get("pass_fail") == "fail"

    @pytest.mark.asyncio
    async def test_map50_none_when_no_result_json_and_no_log(self, tmp_path):
        """沒有 result.json 且 log 為空 → map50=None, pass_fail=None（不應 crash）"""
        import pollers.kaggle_poller as kp
        importlib.reload(kp)

        sub = _make_sub(map50_threshold=0.6)
        db = MagicMock()
        mv_kwargs: dict = {}

        class CapturingMV:
            def __init__(self, **kwargs):
                mv_kwargs.update(kwargs)

        with patch.object(kp, "_download_kernel_output",
                          new=AsyncMock(return_value=str(tmp_path))), \
             patch.object(kp, "_read_log_files", return_value=""), \
             patch.object(kp, "_next_version_for", return_value="v1"), \
             patch.object(kp, "_append_history"), \
             patch.object(kp, "notify_event", new=AsyncMock()), \
             patch.object(kp, "ModelVersion", CapturingMV):

            await kp._on_kernel_complete(db, sub)

        assert mv_kwargs.get("map50") is None
        assert mv_kwargs.get("pass_fail") is None


# ---------------------------------------------------------------------------
# 6. PPE log 格式（MH-2026-018）— 超大 NDJSON log 不被截斷
# ---------------------------------------------------------------------------

# PPE kernel 的 result.json 沒有輸出為獨立檔案，只 print() 到 stdout，
# 所以 NDJSON log 是唯一的 metrics 來源。
# 重現 bug：原 read_log_files 5MB 上限將 5.5MB log 截斷為空字串。

SAMPLE_PPE_NDJSON_LOG = (
    '[{"stream_name":"stdout","time":1.0,"data":"[INIT] req=MH-2026-018 epochs=200\\n"}\n'
    ',{"stream_name":"stdout","time":2.0,"data":"Epoch 178/200\\n"}\n'
    ',{"stream_name":"stdout","time":3.0,"data":"                   all       130      1800      0.920      0.870      0.895      0.586\\n"}\n'
    ',{"stream_name":"stdout","time":4.0,"data":"              person       130       780      0.911      0.868      0.889      0.578\\n"}\n'
    ',{"stream_name":"stdout","time":5.0,"data":"             hardhat       130       420      0.935      0.920      0.942      0.620\\n"}\n'
    ',{"stream_name":"stdout","time":6.0,"data":"          no_hardhat       130       240      0.870      0.840      0.829      0.533\\n"}\n'
    ',{"stream_name":"stdout","time":7.0,"data":"mAP50=0.8948  mAP50-95=0.5861\\n"}\n'
    ',{"stream_name":"stdout","time":8.0,"data":"[PER-CLASS mAP50] {\\"person\\": 0.8893, \\"hardhat\\": 0.9424}\\n"}\n'
    ']'
)


class TestPPELogFormat:
    """MH-2026-018 PPE 格式測試（NDJSON，無獨立 result.json）"""

    def test_ppe_ndjson_parse_yields_map50(self):
        """PPE NDJSON log → parse_yolo_log 應能取得 map50"""
        yolo_mod = _import_parsers_yolo()
        result = yolo_mod.parse_yolo_log(SAMPLE_PPE_NDJSON_LOG)
        assert result["metrics"].get("map50") is not None
        assert result["metrics"]["map50"] == pytest.approx(0.895, abs=0.01)
        assert result["metrics"]["map50_95"] == pytest.approx(0.586, abs=0.01)

    def test_ppe_per_class_excludes_all_key(self):
        """per_class dict 不應包含 'all' 這個 key"""
        yolo_mod = _import_parsers_yolo()
        result = yolo_mod.parse_yolo_log(SAMPLE_PPE_NDJSON_LOG)
        per_class = result.get("per_class") or {}
        assert "all" not in per_class, f"'all' 不應在 per_class 中，實際：{per_class}"

    def test_ppe_per_class_contains_ppe_classes(self):
        """per_class 包含正確的 PPE class 名稱"""
        yolo_mod = _import_parsers_yolo()
        result = yolo_mod.parse_yolo_log(SAMPLE_PPE_NDJSON_LOG)
        per_class = result.get("per_class") or {}
        # PPE kernel 含 person / hardhat / no_hardhat
        assert "person" in per_class or "hardhat" in per_class, \
            f"應含 PPE class，實際：{per_class}"

    def test_read_log_files_reads_large_log(self, tmp_path):
        """read_log_files 應能讀取超過舊 5MB 上限的 log 檔（修復 bug）"""
        import importlib
        import utils as utils_mod
        importlib.reload(utils_mod)

        # 製造一個 6MB log 檔（超過舊 5MB 限制）
        large_log = tmp_path / "train.log"
        # PPE-like NDJSON 重複到 6MB
        chunk = (
            ',{"stream_name":"stdout","time":1.0,"data":"Epoch 1/200\\n"}\n'
            * 10  # ~1KB
        )
        content = "[" + chunk[1:] + "]"  # 移除第一個逗號
        # 重複到約 6MB
        repeats = (6 * 1024 * 1024) // len(content.encode()) + 1
        final_content = "[" + (chunk * repeats) + "]"
        large_log.write_text(final_content)
        size_mb = large_log.stat().st_size / 1024 / 1024
        assert size_mb > 5.0, f"測試 log 應大於 5MB，實際 {size_mb:.1f}MB"

        result = utils_mod.read_log_files(str(tmp_path))
        assert len(result) > 0, "read_log_files 不應回傳空字串（修復 5MB 截斷 bug）"
        assert "Epoch" in result

    def test_ppe_pass_fail_above_07_threshold(self):
        """PPE mAP50=0.895 > threshold 0.7 → pass"""
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        assert kp._compute_pass_fail(0.895, 0.7) == "pass"

    def test_ppe_fail_below_07_threshold(self):
        """mAP50=0.65 < threshold 0.7 → fail"""
        import pollers.kaggle_poller as kp
        importlib.reload(kp)
        assert kp._compute_pass_fail(0.65, 0.7) == "fail"
