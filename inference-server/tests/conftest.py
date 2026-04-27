"""
conftest.py — inference-server 測試環境

Mock 所有重 ML 套件（ultralytics / transformers / sentence_transformers / xgboost / joblib），
讓 CI 不需要 GPU 也能跑測試。
"""
import sys
import types
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# 預先 mock 重 ML 套件（防止 import 時爆炸）
# ---------------------------------------------------------------------------

def _make_mock_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# ultralytics
if "ultralytics" not in sys.modules:
    _yolo_mock = mock.MagicMock()
    sys.modules["ultralytics"] = _make_mock_module("ultralytics", YOLO=_yolo_mock)

# transformers
if "transformers" not in sys.modules:
    _tok_mock = mock.MagicMock()
    _model_mock = mock.MagicMock()
    sys.modules["transformers"] = _make_mock_module(
        "transformers",
        AutoTokenizer=_tok_mock,
        AutoModelForCausalLM=_model_mock,
    )

# sentence_transformers
if "sentence_transformers" not in sys.modules:
    _st_mock = mock.MagicMock()
    sys.modules["sentence_transformers"] = _make_mock_module(
        "sentence_transformers",
        SentenceTransformer=_st_mock,
    )

# joblib
if "joblib" not in sys.modules:
    sys.modules["joblib"] = _make_mock_module("joblib", load=mock.MagicMock())

# xgboost (optional, not directly imported by main)
if "xgboost" not in sys.modules:
    sys.modules["xgboost"] = _make_mock_module("xgboost")

# torch（sentence-transformer / llamaguard 需要）
if "torch" not in sys.modules:
    _torch_mock = _make_mock_module(
        "torch",
        no_grad=mock.MagicMock(return_value=mock.MagicMock(__enter__=lambda s: s, __exit__=mock.MagicMock(return_value=False))),
    )
    sys.modules["torch"] = _torch_mock

# Pillow
if "PIL" not in sys.modules:
    _pil_image_mock = mock.MagicMock()
    _pil_mock = _make_mock_module("PIL", Image=_pil_image_mock)
    sys.modules["PIL"] = _pil_mock
    sys.modules["PIL.Image"] = _pil_image_mock

# prometheus_client
if "prometheus_client" not in sys.modules:
    _counter_mock = mock.MagicMock()
    _histogram_mock = mock.MagicMock()
    _gauge_mock = mock.MagicMock()
    sys.modules["prometheus_client"] = _make_mock_module(
        "prometheus_client",
        Counter=mock.MagicMock(return_value=_counter_mock),
        Histogram=mock.MagicMock(return_value=_histogram_mock),
        Gauge=mock.MagicMock(return_value=_gauge_mock),
        generate_latest=mock.MagicMock(return_value=b"# test metrics\n"),
        CONTENT_TYPE_LATEST="text/plain; version=0.0.4",
    )

# apscheduler
if "apscheduler" not in sys.modules:
    _sched_mock = mock.MagicMock()
    _bg_sched = mock.MagicMock()
    sys.modules["apscheduler"] = _make_mock_module("apscheduler")
    sys.modules["apscheduler.schedulers"] = _make_mock_module("apscheduler.schedulers")
    sys.modules["apscheduler.schedulers.background"] = _make_mock_module(
        "apscheduler.schedulers.background",
        BackgroundScheduler=mock.MagicMock(return_value=_bg_sched),
    )
