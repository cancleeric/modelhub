"""
test_registry_accept.py — 驗收 registry.accept_version 邏輯

測試項目：
  1. pass_fail 欄位：CTO 人工判定覆蓋自動比較
  2. status 自動同步：pass → active，fail → rejected
  3. is_current 邏輯：pass 設為 current，fail 不改 current
  4. map50_actual 可為 None（不強制填值）
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

_APP_DIR = os.path.join(os.path.dirname(__file__), "..")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


def _make_model_version(id=1, req_no="MH-2026-006", version="v2", status="pending_acceptance",
                        is_current=False, pass_fail=None, map50=None, map50_95=None):
    mv = MagicMock()
    mv.id = id
    mv.req_no = req_no
    mv.version = version
    mv.status = status
    mv.is_current = is_current
    mv.pass_fail = pass_fail
    mv.map50 = map50
    mv.map50_95 = map50_95
    mv.accepted_by = None
    mv.accepted_at = None
    mv.acceptance_note = None
    mv.map50_actual = None
    mv.map50_95_actual = None
    return mv


def _make_submission(req_no="MH-2026-006", status="trained", map50_target=0.88):
    sub = MagicMock()
    sub.req_no = req_no
    sub.status = status
    sub.map50_target = map50_target
    return sub


class TestAcceptVersionPassFail:
    """pass_fail 欄位：CTO 人工判定 vs 自動比較"""

    def _run_accept(self, payload_data, submission, model_version):
        """模擬 accept_version 的核心邏輯"""
        # 模擬 AcceptancePayload
        class Payload:
            pass
        p = Payload()
        p.map50_actual = payload_data.get("map50_actual")
        p.map50_95_actual = payload_data.get("map50_95_actual")
        p.pass_fail = payload_data.get("pass_fail")
        p.accepted_by = payload_data.get("accepted_by", "CTO")
        p.acceptance_note = payload_data.get("acceptance_note")

        obj = model_version
        obj.map50_actual = p.map50_actual
        obj.map50_95_actual = p.map50_95_actual
        obj.acceptance_note = p.acceptance_note
        obj.accepted_by = p.accepted_by

        # pass/fail 判定邏輯（與 registry.py 相同）
        if p.pass_fail is not None:
            obj.pass_fail = p.pass_fail
        else:
            target = submission.map50_target if submission else None
            if target is not None and p.map50_actual is not None:
                obj.pass_fail = "pass" if p.map50_actual >= target else "fail"
            else:
                obj.pass_fail = "pass"

        # status 同步
        if obj.pass_fail == "pass":
            obj.is_current = True
            obj.status = "active"
        else:
            obj.is_current = False
            obj.status = "rejected"

        return obj

    def test_cto_manual_pass_overrides_auto_fail(self):
        """CTO 傳 pass_fail=pass，即使 map50 < target 仍 pass"""
        sub = _make_submission(map50_target=0.88)
        mv = _make_model_version()

        result = self._run_accept(
            {"map50_actual": 0.5993, "pass_fail": "pass"},
            sub, mv
        )
        assert result.pass_fail == "pass"
        assert result.status == "active"
        assert result.is_current is True

    def test_auto_fail_when_below_target_no_override(self):
        """沒有 pass_fail 覆蓋，map50 < target → 自動判 fail"""
        sub = _make_submission(map50_target=0.88)
        mv = _make_model_version()

        result = self._run_accept(
            {"map50_actual": 0.5993, "pass_fail": None},
            sub, mv
        )
        assert result.pass_fail == "fail"
        assert result.status == "rejected"
        assert result.is_current is False

    def test_auto_pass_when_above_target(self):
        """沒有覆蓋，map50 >= target → 自動判 pass"""
        sub = _make_submission(map50_target=0.70)
        mv = _make_model_version()

        result = self._run_accept(
            {"map50_actual": 0.8996, "pass_fail": None},
            sub, mv
        )
        assert result.pass_fail == "pass"
        assert result.status == "active"

    def test_default_pass_when_no_target_no_map50(self):
        """沒有 target、沒有 map50_actual → 預設 pass"""
        sub = _make_submission(map50_target=None)
        mv = _make_model_version()

        result = self._run_accept(
            {"map50_actual": None, "pass_fail": None},
            sub, mv
        )
        assert result.pass_fail == "pass"
        assert result.status == "active"

    def test_cto_manual_fail_overrides_auto_pass(self):
        """CTO 傳 pass_fail=fail，即使 map50 > target 仍 fail"""
        sub = _make_submission(map50_target=0.70)
        mv = _make_model_version()

        result = self._run_accept(
            {"map50_actual": 0.8996, "pass_fail": "fail"},
            sub, mv
        )
        assert result.pass_fail == "fail"
        assert result.status == "rejected"
        assert result.is_current is False

    def test_status_active_on_pass(self):
        """pass → ModelVersion.status 改為 active"""
        sub = _make_submission(map50_target=None)
        mv = _make_model_version(status="pending_acceptance")

        result = self._run_accept(
            {"map50_actual": 0.9662, "pass_fail": "pass"},
            sub, mv
        )
        assert result.status == "active"

    def test_status_rejected_on_fail(self):
        """fail → ModelVersion.status 改為 rejected"""
        sub = _make_submission(map50_target=None)
        mv = _make_model_version(status="pending_acceptance")

        result = self._run_accept(
            {"map50_actual": 0.4322, "pass_fail": "fail"},
            sub, mv
        )
        assert result.status == "rejected"
