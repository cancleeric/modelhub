"""
conftest.py — 測試環境 setup

解決 Python 3.9 不支援 str | None 語法問題（parsers/__init__.py 使用了 3.10+ 語法）。
在 pytest 收集前預先 mock 有問題的模組。
"""
import sys
import types
import unittest.mock as mock


def _install_mocks():
    """安裝所有需要 mock 的模組（在 import 前執行）"""
    # mock parsers（parsers/__init__.py 使用 str | None 語法，Python 3.9 不支援）
    if 'parsers' not in sys.modules:
        parsers_mock = types.ModuleType('parsers')
        parsers_mock.parse_training_log = lambda arch, log_text: {"metrics": {}, "per_class": {}}
        sys.modules['parsers'] = parsers_mock

    # mock models（需要 DB engine 連線）
    if 'models' not in sys.modules:
        models_mock = mock.MagicMock()
        # 保留真正需要的類型供測試用
        models_mock.SessionLocal = mock.MagicMock()
        models_mock.Submission = mock.MagicMock()
        models_mock.ModelVersion = mock.MagicMock()
        models_mock.SubmissionHistory = mock.MagicMock()
        models_mock.TrainingQueue = mock.MagicMock()
        sys.modules['models'] = models_mock

    # mock notifications
    if 'notifications' not in sys.modules:
        notif_mock = types.ModuleType('notifications')
        notif_mock.notify = mock.AsyncMock(return_value=True)
        notif_mock.notify_event = mock.AsyncMock(return_value=None)
        notif_mock.CTO_TARGET = "cto@hurricanecore.internal"
        sys.modules['notifications'] = notif_mock


_install_mocks()
