"""
M16 unit tests — epoch_curve 解析功能

測試範圍：
- parse_results_csv: YOLOv8 results.csv 逐 epoch 解析
- _parse_epoch_data_from_stdout: stdout 格式 fallback 解析
- SubmissionOut schema 含 epoch_curve 欄位

注意：conftest.py 把 parsers mock 掉，所以這裡使用 importlib.util 直接載入
parsers/yolo.py 的真實實作，繞過 mock。
"""
import importlib.util
import os
import sys

import pytest

# 直接載入 parsers/yolo.py（繞過 conftest 的 mock）
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YOLO_SPEC = importlib.util.spec_from_file_location(
    "_yolo_real",
    os.path.join(_BACKEND_DIR, "parsers", "yolo.py"),
)
_yolo = importlib.util.module_from_spec(_YOLO_SPEC)  # type: ignore
_YOLO_SPEC.loader.exec_module(_yolo)  # type: ignore

parse_results_csv = _yolo.parse_results_csv
_parse_epoch_data_from_stdout = _yolo._parse_epoch_data_from_stdout
_parse_yolo_log_real = _yolo.parse_yolo_log


# ── results.csv 測試 ───────────────────────────────────────────

SAMPLE_RESULTS_CSV = """\
                   epoch,         train/box_loss,         train/cls_loss,         train/dfl_loss,   metrics/precision(B),      metrics/recall(B),       metrics/mAP50(B),    metrics/mAP50-95(B),           val/box_loss,           val/cls_loss,           val/dfl_loss,                 lr/pg0,                 lr/pg1,                 lr/pg2
                       0,                1.2345,                0.8901,                1.1111,                0.1234,                0.2345,                0.3456,                0.1234,                0.9876,                0.5678,                0.8765,              0.001000,              0.001000,              0.001000
                       1,                0.9876,                0.7654,                1.0000,                0.2345,                0.3456,                0.4567,                0.2123,                0.8765,                0.4567,                0.7654,              0.001000,              0.001000,              0.001000
                       2,                0.8765,                0.6543,                0.9999,                0.4567,                0.5678,                0.6789,                0.3456,                0.7654,                0.3456,                0.6543,              0.000900,              0.000900,              0.000900
"""


def test_parse_results_csv_basic():
    data = parse_results_csv(SAMPLE_RESULTS_CSV)
    assert len(data) == 3
    first = data[0]
    assert first["epoch"] == 0
    assert "train_loss" in first
    assert "val_loss" in first
    assert "map50" in first
    assert "map50_95" in first
    # train_loss = box + cls = 1.2345 + 0.8901
    assert abs(first["train_loss"] - (1.2345 + 0.8901)) < 1e-4
    assert abs(first["map50"] - 0.3456) < 1e-4


def test_parse_results_csv_monotone_map50():
    """map50 應遞增（在此 sample 中）"""
    data = parse_results_csv(SAMPLE_RESULTS_CSV)
    map50s = [d["map50"] for d in data]
    assert map50s == sorted(map50s)


def test_parse_results_csv_empty():
    assert parse_results_csv("") == []


def test_parse_results_csv_header_only():
    header = "epoch,train/box_loss,train/cls_loss,metrics/mAP50(B)\n"
    result = parse_results_csv(header)
    assert result == []


def test_parse_results_csv_with_spaces():
    """欄位名前後有空格（YOLOv8 實際輸出）"""
    csv_text = "  epoch  ,  train/box_loss  ,  metrics/mAP50(B)  \n  0  ,  1.0  ,  0.5  \n"
    data = parse_results_csv(csv_text)
    assert len(data) == 1
    assert data[0]["epoch"] == 0
    assert data[0]["map50"] == pytest.approx(0.5, abs=1e-6)


# ── stdout fallback 測試 ───────────────────────────────────────

SAMPLE_STDOUT = """\
Epoch 1/3: Training...
      Class     Images  Instances          P          R      mAP50   mAP50-95
        all        100        500       0.82       0.75      0.423      0.195
Epoch 2/3: Training...
      Class     Images  Instances          P          R      mAP50   mAP50-95
        all        100        500       0.85       0.78      0.512      0.234
Epoch 3/3: Training...
      Class     Images  Instances          P          R      mAP50   mAP50-95
        all        100        500       0.88       0.81      0.601      0.278
"""


def test_parse_epoch_data_from_stdout_basic():
    data = _parse_epoch_data_from_stdout(SAMPLE_STDOUT)
    assert len(data) == 3
    assert data[0]["epoch"] == 1
    assert data[2]["epoch"] == 3
    assert data[2]["map50"] == pytest.approx(0.601, abs=1e-4)


def test_parse_epoch_data_from_stdout_empty():
    assert _parse_epoch_data_from_stdout("") == []


# ── parse_yolo_log 整合測試（含 epoch_data 欄位）─────────────────

def test_parse_yolo_log_includes_epoch_data_key():
    """parse_yolo_log 回傳值必須含 epoch_data 欄位（M16 新增）"""
    result = _parse_yolo_log_real("no log content")
    assert "epoch_data" in result


def test_parse_yolo_log_result_json_path_epoch_data_empty():
    """
    result.json 路徑（_source=result_json）的 epoch_data 必須是空 list
    （逐 epoch 資料從 results.csv 補充，不在 result.json 路徑中）
    """
    result_json_line = '{"map50": 0.7455, "map50_95": 0.5, "epochs": 50}'
    result = _parse_yolo_log_real(result_json_line)
    assert result.get("_source") == "result_json"
    assert result["epoch_data"] == []


def test_parse_yolo_log_stdout_path_epoch_data():
    """stdout 解析路徑應回傳 epoch_data key"""
    result = _parse_yolo_log_real(SAMPLE_STDOUT)
    assert "epoch_data" in result


# ── SubmissionOut schema 含 epoch_curve 欄位 ─────────────────

def test_submission_out_has_epoch_curve_field():
    """SubmissionOut schema 必須有 epoch_curve Optional[str] 欄位"""
    # 因 conftest mock 了 models，這裡直接 import routers.submissions（依賴 models mock）
    from routers.submissions import SubmissionOut  # type: ignore[import]
    fields = SubmissionOut.model_fields
    assert "epoch_curve" in fields
    # 預設值應為 None（optional）
    info = fields["epoch_curve"]
    assert info.default is None or info.is_required() is False
