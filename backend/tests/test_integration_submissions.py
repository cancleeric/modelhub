"""
test_integration_submissions.py — submissions + actions 整合測試

使用 SQLite in-memory DB，跑完整 submit → approve → start_training → trained → accept 流程。
不依賴 conftest.py mock（使用自己的 mini FastAPI app + fixture）。
"""

import sys
import types
import unittest.mock as mock
import pytest
import os

os.environ.setdefault("SKIP_ROLE_CHECK", "true")
os.environ.setdefault("AUTO_APPROVE_AFTER_VALIDATOR", "false")
os.environ["DATABASE_URL"] = "sqlite://"

# ---------------------------------------------------------------------------
# 在 module-level 補齊 mock（conftest 可能已設定 models 為 MagicMock，需重設）
# ---------------------------------------------------------------------------
if "models" in sys.modules and isinstance(sys.modules["models"], mock.MagicMock):
    del sys.modules["models"]

# 確保 notifications 是 mock
if "notifications" not in sys.modules or not hasattr(sys.modules["notifications"], "notify_event"):
    notif = types.ModuleType("notifications")
    notif.notify = mock.AsyncMock(return_value=True)
    notif.notify_event = mock.AsyncMock(return_value=None)
    notif.CTO_TARGET = "cto@test"
    sys.modules["notifications"] = notif
else:
    # 確保 notify_event 是 AsyncMock
    sys.modules["notifications"].notify_event = mock.AsyncMock(return_value=None)

# 確保 validators 是 mock
if "validators" not in sys.modules:
    vm = types.ModuleType("validators")
    async def _noop_validate(payload): return []
    vm.validate_submission = _noop_validate
    sys.modules["validators"] = vm

# 確保 advisors.llm_advisor 是 mock
if "advisors" not in sys.modules:
    sys.modules["advisors"] = types.ModuleType("advisors")
if "advisors.llm_advisor" not in sys.modules:
    llm_adv = types.ModuleType("advisors.llm_advisor")
    async def _noop_review(payload): return []
    llm_adv.review_submission = _noop_review
    sys.modules["advisors.llm_advisor"] = llm_adv

# 確保 parsers 是 mock
if "parsers" not in sys.modules:
    pm = types.ModuleType("parsers")
    pm.parse_training_log = lambda arch, log_text: {"metrics": {}, "per_class": {}}
    sys.modules["parsers"] = pm

import models as _models_module
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch as _mock_patch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def db_engine():
    """
    每個測試用獨立 in-memory SQLite engine。
    使用 StaticPool 確保所有 connection 共用同一個底層連線（in-memory DB 不跨連線）。
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _models_module.Base.metadata.create_all(bind=engine)
    yield engine
    _models_module.Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(db_engine):
    """每個測試用獨立 Session"""
    Session = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="function")
def app_client(db_engine):
    """
    建立一個 mini FastAPI app（只含 submissions + actions router），
    使用 in-memory SQLite。不啟動任何 scheduler / lifespan。
    """
    from fastapi import FastAPI as _FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from models import get_db
    import auth as _auth_module
    from auth import get_current_user, get_current_user_or_api_key
    import routers.submissions as _sub_router
    import routers.actions as _act_router

    # 確保 SKIP_ROLE_CHECK=true，避免測試間污染
    _orig_skip = _auth_module._SKIP_ROLE_CHECK
    _auth_module._SKIP_ROLE_CHECK = True

    test_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def override_get_db():
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    async def override_current_user():
        return {"sub": "test-user", "preferred_username": "tester", "modelhub_role": "reviewer"}

    async def override_current_user_or_api_key():
        return {"sub": "test-user", "preferred_username": "tester", "modelhub_role": "reviewer"}

    def _noop_resource_handler(obj, db):
        obj.training_resource = "local_mps"

    mini_app = _FastAPI()
    mini_app.include_router(_sub_router.router, prefix="/api/submissions")
    mini_app.include_router(_act_router.router, prefix="/api/submissions")

    mini_app.dependency_overrides[get_db] = override_get_db
    mini_app.dependency_overrides[get_current_user] = override_current_user
    mini_app.dependency_overrides[get_current_user_or_api_key] = override_current_user_or_api_key

    with _mock_patch("routers.actions._handle_start_training_resource", side_effect=_noop_resource_handler):
        with TestClient(mini_app, raise_server_exceptions=True) as client:
            yield client

    _auth_module._SKIP_ROLE_CHECK = _orig_skip


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

MINIMAL_SUBMISSION = {
    "product": "AICAD",
    "company": "HurricaneEdge",
    "submitter": "tester",
    "priority": "P2",
}


def create_and_get_req_no(client) -> str:
    resp = client.post("/api/submissions/", json=MINIMAL_SUBMISSION)
    assert resp.status_code == 201, f"create failed: {resp.text}"
    return resp.json()["submission"]["req_no"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubmissionStateMachine:

    def test_create_submission_draft(self, app_client):
        """POST /api/submissions/ → status=draft"""
        resp = app_client.post("/api/submissions/", json=MINIMAL_SUBMISSION)
        assert resp.status_code == 201
        body = resp.json()
        assert body["submission"]["status"] == "draft"
        assert body["submission"]["req_no"].startswith("MH-")

    def test_submit_action(self, app_client):
        """draft → submit → submitted"""
        req_no = create_and_get_req_no(app_client)
        resp = app_client.post(f"/api/submissions/{req_no}/actions/submit", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "submitted"

    def test_approve_action(self, app_client):
        """submitted → approve → approved"""
        req_no = create_and_get_req_no(app_client)
        app_client.post(f"/api/submissions/{req_no}/actions/submit", json={})
        resp = app_client.post(f"/api/submissions/{req_no}/actions/approve", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    def test_start_training_action(self, app_client):
        """approved → start_training → training"""
        req_no = create_and_get_req_no(app_client)
        app_client.post(f"/api/submissions/{req_no}/actions/submit", json={})
        app_client.post(f"/api/submissions/{req_no}/actions/approve", json={})
        resp = app_client.post(f"/api/submissions/{req_no}/actions/start_training", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "training"

    def test_training_result_trained(self, app_client):
        """training → PATCH training-result status=trained → trained"""
        req_no = create_and_get_req_no(app_client)
        app_client.post(f"/api/submissions/{req_no}/actions/submit", json={})
        app_client.post(f"/api/submissions/{req_no}/actions/approve", json={})
        app_client.post(f"/api/submissions/{req_no}/actions/start_training", json={})
        resp = app_client.patch(
            f"/api/submissions/{req_no}/training-result",
            json={"status": "trained", "metrics": {"map50": 0.72, "epochs": 50}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "trained"

    def test_accept_action(self, app_client):
        """trained → accept → accepted（完整 happy path）"""
        req_no = create_and_get_req_no(app_client)
        app_client.post(f"/api/submissions/{req_no}/actions/submit", json={})
        app_client.post(f"/api/submissions/{req_no}/actions/approve", json={})
        app_client.post(f"/api/submissions/{req_no}/actions/start_training", json={})
        app_client.patch(
            f"/api/submissions/{req_no}/training-result",
            json={"status": "trained"},
        )
        resp = app_client.post(f"/api/submissions/{req_no}/actions/accept", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_state_machine_whitelist_blocks_invalid_transition(self, app_client):
        """draft 狀態不能直接 approve（狀態機保護）"""
        req_no = create_and_get_req_no(app_client)
        resp = app_client.post(f"/api/submissions/{req_no}/actions/approve", json={})
        assert resp.status_code == 422

    def test_patch_blocks_core_fields_in_training(self, app_client):
        """training 狀態禁止修改 arch（PATCH 狀態機白名單）"""
        req_no = create_and_get_req_no(app_client)
        app_client.post(f"/api/submissions/{req_no}/actions/submit", json={})
        app_client.post(f"/api/submissions/{req_no}/actions/approve", json={})
        app_client.post(f"/api/submissions/{req_no}/actions/start_training", json={})
        resp = app_client.patch(f"/api/submissions/{req_no}", json={"arch": "yolov8n"})
        assert resp.status_code == 422

    def test_reject_and_resubmit(self, app_client):
        """submitted → reject → rejected → resubmit → submitted"""
        req_no = create_and_get_req_no(app_client)
        app_client.post(f"/api/submissions/{req_no}/actions/submit", json={})
        resp = app_client.post(
            f"/api/submissions/{req_no}/reject",
            json={"reasons": ["dataset_insufficient"], "note": "測試退件"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

        resp = app_client.post(
            f"/api/submissions/{req_no}/resubmit",
            json={"note": "補件完成"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "submitted"

    def test_get_submission(self, app_client):
        """GET /api/submissions/{req_no} → 返回 SubmissionOut"""
        req_no = create_and_get_req_no(app_client)
        resp = app_client.get(f"/api/submissions/{req_no}")
        assert resp.status_code == 200
        assert resp.json()["req_no"] == req_no

    def test_list_submissions(self, app_client):
        """GET /api/submissions/ → 返回 200（list API 正常回應）"""
        create_and_get_req_no(app_client)
        resp = app_client.get("/api/submissions/")
        # 注意：list 使用 req_no LIKE 過濾 magic string，正常 req_no 不受影響
        assert resp.status_code == 200
        # 至少 API 可呼叫（items 數量依 filter 行為而定）
        assert isinstance(resp.json(), list)

    def test_training_result_invalid_status(self, app_client):
        """PATCH training-result 給非法 status → 422"""
        req_no = create_and_get_req_no(app_client)
        app_client.post(f"/api/submissions/{req_no}/actions/submit", json={})
        app_client.post(f"/api/submissions/{req_no}/actions/approve", json={})
        app_client.post(f"/api/submissions/{req_no}/actions/start_training", json={})
        resp = app_client.patch(
            f"/api/submissions/{req_no}/training-result",
            json={"status": "bogus_status"},
        )
        assert resp.status_code == 422

    def test_req_no_sequential(self, app_client):
        """多張工單 req_no 應遞增"""
        r1 = create_and_get_req_no(app_client)
        r2 = create_and_get_req_no(app_client)
        seq1 = int(r1.split("-")[-1])
        seq2 = int(r2.split("-")[-1])
        assert seq2 == seq1 + 1
