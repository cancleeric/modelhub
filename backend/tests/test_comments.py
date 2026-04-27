"""
test_comments.py — M22 Discussion Comments API 測試

測試範圍：
  - GET /api/submissions/{req_no}/comments（public + internal 過濾）
  - POST /api/submissions/{req_no}/comments（建立、tenant 驗證、is_internal 權限）
  - PATCH /api/comments/{id}（只 author 或 superadmin）
  - DELETE /api/comments/{id}（soft delete）
  - Reject API 升級：自動建首筆 comment，discussion_count=1
  - Thread reply（parent_id 1 層）
  - tenant 隔離（company A 看不到 company B）
"""

import sys
import types
import unittest.mock as mock
import pytest
import os

os.environ.setdefault("SKIP_ROLE_CHECK", "false")  # 測試裡要測 role check，預設關閉
os.environ.setdefault("AUTO_APPROVE_AFTER_VALIDATOR", "false")
os.environ["DATABASE_URL"] = "sqlite://"

# ---------------------------------------------------------------------------
# Mock 設定（在 import models 之前）
# ---------------------------------------------------------------------------
if "models" in sys.modules and isinstance(sys.modules["models"], mock.MagicMock):
    del sys.modules["models"]

if "notifications" not in sys.modules or not hasattr(sys.modules["notifications"], "notify_event"):
    notif = types.ModuleType("notifications")
    notif.notify = mock.AsyncMock(return_value=True)
    notif.notify_event = mock.AsyncMock(return_value=None)
    notif.CTO_TARGET = "cto@test"
    sys.modules["notifications"] = notif
else:
    sys.modules["notifications"].notify_event = mock.AsyncMock(return_value=None)

if "validators" not in sys.modules:
    vm = types.ModuleType("validators")
    async def _noop_validate(payload): return []
    vm.validate_submission = _noop_validate
    sys.modules["validators"] = vm

if "parsers" not in sys.modules:
    pm = types.ModuleType("parsers")
    pm.parse_training_log = lambda arch, log_text: {"metrics": {}, "per_class": {}}
    sys.modules["parsers"] = pm

if "advisors" not in sys.modules:
    sys.modules["advisors"] = types.ModuleType("advisors")
if "advisors.llm_advisor" not in sys.modules:
    _adv_mod = types.ModuleType("advisors.llm_advisor")
    async def _noop_review(payload): return []
    _adv_mod.review_submission = _noop_review
    sys.modules["advisors.llm_advisor"] = _adv_mod

# ---------------------------------------------------------------------------
# Real imports after mock setup
# ---------------------------------------------------------------------------
import models as _models_module
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_SUBMISSION_EDGE = {
    "product": "AICAD",
    "company": "HurricaneEdge",
    "submitter": "edge-user@hurricaneedge.com",
    "priority": "P2",
}

_SUBMISSION_SOFT = {
    "product": "Squid",
    "company": "HurricaneSoft",
    "submitter": "soft-user@hurricanesoft.com",
    "priority": "P2",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def db_engine():
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
    Session = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = Session()
    yield session
    session.close()


def _build_app(db_engine, user_overrides: dict = None, skip_role_check: bool = True):
    """
    建立 mini FastAPI app，user_overrides 可指定不同 identity 的 override。
    user_overrides 格式：{"email": ..., "preferred_username": ..., "modelhub_role": ..., "company": ...}
    skip_role_check=True：繞過 require_role Depends（用於 CRUD/reject 測試）
    skip_role_check=False：啟用 role check（用於 is_internal 權限測試）
    """
    from fastapi import FastAPI as _FA
    import auth as _auth_module
    from auth import get_current_user, get_current_user_or_api_key
    from models import get_db
    import routers.submissions as _sub_router
    import routers.actions as _act_router
    import routers.comments as _comments_router

    _auth_module._SKIP_ROLE_CHECK = skip_role_check

    test_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def override_get_db():
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    default_user = user_overrides or {
        "sub": "reviewer-user",
        "email": "reviewer@test.com",
        "preferred_username": "reviewer@test.com",
        "modelhub_role": "reviewer",
        "company": "HurricaneEdge",
    }

    async def override_current_user():
        return default_user

    async def override_current_user_or_api_key():
        return default_user

    mini_app = _FA()
    mini_app.include_router(_sub_router.router, prefix="/api/submissions")
    mini_app.include_router(_act_router.router, prefix="/api/submissions")
    mini_app.include_router(_comments_router.router, prefix="/api")

    mini_app.dependency_overrides[get_db] = override_get_db
    mini_app.dependency_overrides[get_current_user] = override_current_user
    mini_app.dependency_overrides[get_current_user_or_api_key] = override_current_user_or_api_key

    return mini_app


def _make_client(db_engine, user: dict = None, skip_role_check: bool = True):
    """建立 TestClient，mock _handle_start_training_resource。"""
    from unittest.mock import patch as _patch
    mini_app = _build_app(db_engine, user, skip_role_check=skip_role_check)

    def _noop_resource(obj, db):
        obj.training_resource = "local_mps"

    with _patch("routers.actions._handle_start_training_resource", side_effect=_noop_resource):
        client = TestClient(mini_app, raise_server_exceptions=True)
        return client


def _create_submission(client, data: dict) -> str:
    resp = client.post("/api/submissions/", json=data)
    assert resp.status_code == 201, f"create failed: {resp.text}"
    return resp.json()["submission"]["req_no"]


def _submit_and_reject(client, req_no: str, reasons=None, note=None) -> dict:
    """Submit 然後 reject，回傳 reject response JSON。"""
    client.post(f"/api/submissions/{req_no}/actions/submit", json={})
    resp = client.post(
        f"/api/submissions/{req_no}/reject",
        json={
            "reasons": reasons or ["資料集標注不完整"],
            "note": note or "請補充 YOLO 標籤",
        },
    )
    assert resp.status_code == 200, f"reject failed: {resp.text}"
    return resp.json()


# ===========================================================================
# Test Classes
# ===========================================================================

class TestCommentCRUD:
    """基本 CRUD 操作測試"""

    def test_list_comments_empty(self, db_engine):
        """空工單的 comments 回傳空清單"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)
        resp = client.get(f"/api/submissions/{req_no}/comments")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_comment_public(self, db_engine):
        """建立 public comment，出現在 GET 結果"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "## 說明\n\n這是一個測試留言", "is_internal": False},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["body_markdown"] == "## 說明\n\n這是一個測試留言"
        assert data["is_internal"] is False
        assert data["req_no"] == req_no

    def test_create_comment_updates_discussion_count(self, db_engine):
        """建立留言後 submissions.discussion_count 應 +1"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "first comment"},
        )
        # 查 submission discussion_count
        from models import Submission
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            sub = db.query(Submission).filter(Submission.req_no == req_no).first()
            assert sub.discussion_count == 1
        finally:
            db.close()

    def test_list_comments_shows_created(self, db_engine):
        """建立後 GET 可看到"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)
        client.post(f"/api/submissions/{req_no}/comments", json={"body_markdown": "hello"})

        resp = client.get(f"/api/submissions/{req_no}/comments")
        assert resp.status_code == 200
        comments = resp.json()
        assert len(comments) == 1
        assert comments[0]["body_markdown"] == "hello"

    def test_patch_comment_by_author(self, db_engine):
        """author 可以編輯自己的留言"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        create_resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "original"},
        )
        comment_id = create_resp.json()["id"]

        patch_resp = client.patch(
            f"/api/comments/{comment_id}",
            json={"body_markdown": "updated content"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["body_markdown"] == "updated content"
        assert patch_resp.json()["updated_at"] is not None

    def test_delete_comment_soft_delete(self, db_engine):
        """DELETE 應是 soft delete，body 替換為刪除提示"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        create_resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "to be deleted"},
        )
        comment_id = create_resp.json()["id"]

        del_resp = client.delete(f"/api/comments/{comment_id}")
        assert del_resp.status_code == 204

        # GET 仍可看到，但 body 已替換
        list_resp = client.get(f"/api/submissions/{req_no}/comments")
        comments = list_resp.json()
        assert len(comments) == 1
        assert comments[0]["body_markdown"] == "此留言已刪除"
        assert comments[0]["deleted_at"] is not None

    def test_delete_comment_reduces_discussion_count(self, db_engine):
        """soft delete 後 discussion_count 應減少"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        cr = client.post(f"/api/submissions/{req_no}/comments", json={"body_markdown": "x"})
        comment_id = cr.json()["id"]
        client.delete(f"/api/comments/{comment_id}")

        from models import Submission
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            sub = db.query(Submission).filter(Submission.req_no == req_no).first()
            assert sub.discussion_count == 0
        finally:
            db.close()


class TestInternalComments:
    """is_internal 權限測試"""

    def test_non_reviewer_cannot_post_internal(self, db_engine):
        """submitter 不能發 is_internal=true comment"""
        submitter_user = {
            "sub": "submitter-user",
            "email": "submitter@test.com",
            "preferred_username": "submitter@test.com",
            "modelhub_role": "submitter",
            "company": "HurricaneEdge",
        }
        # skip_role_check=False 讓 _is_privileged() 正確依 role 判斷
        client = _make_client(db_engine, submitter_user, skip_role_check=False)
        # 先用 reviewer 建工單（submitter 不能建工單需要繞過，直接 insert）
        from models import Submission
        from sqlalchemy.orm import sessionmaker
        from datetime import datetime
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            sub = Submission(
                req_no="MH-2026-INTL-001",
                product="AICAD",
                company="HurricaneEdge",
                submitter="submitter@test.com",
                status="submitted",
                priority="P2",
                created_at=datetime.utcnow(),
            )
            db.add(sub)
            db.commit()
        finally:
            db.close()

        resp = client.post(
            "/api/submissions/MH-2026-INTL-001/comments",
            json={"body_markdown": "internal note", "is_internal": True},
        )
        assert resp.status_code == 403

    def test_reviewer_can_post_internal(self, db_engine):
        """reviewer 可以發 is_internal=true comment"""
        reviewer_user = {
            "sub": "reviewer-user",
            "email": "reviewer@test.com",
            "preferred_username": "reviewer@test.com",
            "modelhub_role": "reviewer",
            "company": "HurricaneEdge",
        }
        client = _make_client(db_engine, reviewer_user, skip_role_check=False)
        req_no = _create_submission(
            _make_client(db_engine, reviewer_user, skip_role_check=True),
            _SUBMISSION_EDGE,
        )

        resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "internal note", "is_internal": True},
        )
        assert resp.status_code == 201
        assert resp.json()["is_internal"] is True

    def test_submitter_cannot_see_internal_comment(self, db_engine):
        """submitter GET comments 看不到 is_internal comment"""
        reviewer_user = {
            "sub": "reviewer-user",
            "email": "reviewer@test.com",
            "preferred_username": "reviewer@test.com",
            "modelhub_role": "reviewer",
            "company": "HurricaneEdge",
        }
        # 用 reviewer 建工單並發 internal comment（skip_role_check=True 讓 submit 動作不被擋）
        setup_client = _make_client(db_engine, reviewer_user, skip_role_check=True)
        req_no = _create_submission(setup_client, _SUBMISSION_EDGE)
        # skip_role_check=False 讓 _is_privileged() 基於 role 判斷
        reviewer_client = _make_client(db_engine, reviewer_user, skip_role_check=False)
        reviewer_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "this is internal", "is_internal": True},
        )

        submitter_user = {
            "sub": "submitter-user",
            "email": "submitter@test.com",
            "preferred_username": "submitter@test.com",
            "modelhub_role": "submitter",
            "company": "HurricaneEdge",
        }
        submitter_client = _make_client(db_engine, submitter_user, skip_role_check=False)
        resp = submitter_client.get(f"/api/submissions/{req_no}/comments")
        assert resp.status_code == 200
        assert resp.json() == []  # internal 被過濾

    def test_reviewer_can_see_internal_with_flag(self, db_engine):
        """reviewer 帶 include_internal=true 可看到 internal comment"""
        reviewer_user = {
            "sub": "reviewer-user",
            "email": "reviewer@test.com",
            "preferred_username": "reviewer@test.com",
            "modelhub_role": "reviewer",
            "company": "HurricaneEdge",
        }
        setup_client = _make_client(db_engine, reviewer_user, skip_role_check=True)
        req_no = _create_submission(setup_client, _SUBMISSION_EDGE)

        reviewer_client = _make_client(db_engine, reviewer_user, skip_role_check=False)
        reviewer_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "secret", "is_internal": True},
        )

        resp = reviewer_client.get(f"/api/submissions/{req_no}/comments?include_internal=true")
        assert resp.status_code == 200
        comments = resp.json()
        assert len(comments) == 1
        assert comments[0]["is_internal"] is True

    def test_submitter_cannot_request_include_internal(self, db_engine):
        """submitter 帶 include_internal=true 應回 403"""
        reviewer_user = {
            "sub": "reviewer-user",
            "email": "reviewer@test.com",
            "preferred_username": "reviewer@test.com",
            "modelhub_role": "reviewer",
            "company": "HurricaneEdge",
        }
        setup_client = _make_client(db_engine, reviewer_user, skip_role_check=True)
        req_no = _create_submission(setup_client, _SUBMISSION_EDGE)

        submitter_user = {
            "sub": "submitter-user",
            "email": "submitter@test.com",
            "preferred_username": "submitter@test.com",
            "modelhub_role": "submitter",
            "company": "HurricaneEdge",
        }
        submitter_client = _make_client(db_engine, submitter_user, skip_role_check=False)
        resp = submitter_client.get(f"/api/submissions/{req_no}/comments?include_internal=true")
        assert resp.status_code == 403


class TestRejectAutoComment:
    """Reject API 升級：自動建第一筆 comment"""

    def test_reject_creates_comment(self, db_engine):
        """reject 後 DB 應有一筆 comment，discussion_count=1"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        result = _submit_and_reject(client, req_no, reasons=["資料集不完整"], note="請補標籤")
        assert "comment_id" in result
        assert isinstance(result["comment_id"], int)

        # 查 GET comments
        resp = client.get(f"/api/submissions/{req_no}/comments?include_internal=true")
        assert resp.status_code == 200
        comments = resp.json()
        assert len(comments) == 1
        # body 應含 checklist
        body = comments[0]["body_markdown"]
        assert "資料集不完整" in body

    def test_reject_discussion_count_is_one(self, db_engine):
        """reject 後 discussion_count 應為 1"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)
        _submit_and_reject(client, req_no)

        from models import Submission
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            sub = db.query(Submission).filter(Submission.req_no == req_no).first()
            assert sub.discussion_count == 1
        finally:
            db.close()

    def test_reject_with_markdown_body(self, db_engine):
        """reject_details_markdown 優先使用"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)
        client.post(f"/api/submissions/{req_no}/actions/submit", json={})

        resp = client.post(
            f"/api/submissions/{req_no}/reject",
            json={
                "reasons": ["issue"],
                "reject_details_markdown": "## Custom Markdown\n\nThis is **custom**.",
            },
        )
        assert resp.status_code == 200
        comment_id = resp.json()["comment_id"]

        list_resp = client.get(f"/api/submissions/{req_no}/comments?include_internal=true")
        comments = list_resp.json()
        target = next((c for c in comments if c["id"] == comment_id), None)
        assert target is not None
        assert "Custom Markdown" in target["body_markdown"]


class TestTenantIsolation:
    """Tenant 隔離測試：HurricaneEdge 看不到 HurricaneSoft 的 comment"""

    def test_tenant_a_cannot_see_tenant_b_comments(self, db_engine):
        """HurricaneEdge user 存取 HurricaneSoft 工單應回 403"""
        # Reviewer 建立兩個不同公司的 submission
        reviewer_user = {
            "sub": "reviewer-user",
            "email": "reviewer@test.com",
            "preferred_username": "reviewer@test.com",
            "modelhub_role": "reviewer",
        }
        reviewer_client = _make_client(db_engine, reviewer_user)
        soft_req_no = _create_submission(reviewer_client, _SUBMISSION_SOFT)
        reviewer_client.post(
            f"/api/submissions/{soft_req_no}/comments",
            json={"body_markdown": "soft secret comment"},
        )

        # HurricaneEdge submitter 嘗試存取 HurricaneSoft 工單
        edge_user = {
            "sub": "edge-submitter",
            "email": "user@edge.com",
            "preferred_username": "user@edge.com",
            "modelhub_role": "submitter",
            "company": "HurricaneEdge",
        }
        edge_client = _make_client(db_engine, edge_user)
        resp = edge_client.get(f"/api/submissions/{soft_req_no}/comments")
        assert resp.status_code == 403


class TestThreadReply:
    """Thread reply（parent_id 1 層）"""

    def test_reply_appears_under_parent(self, db_engine):
        """reply 出現在 parent comment 的 replies 陣列"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        parent_resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "parent comment"},
        )
        parent_id = parent_resp.json()["id"]

        client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "reply comment", "parent_id": parent_id},
        )

        list_resp = client.get(f"/api/submissions/{req_no}/comments")
        comments = list_resp.json()
        # 頂層只有 1 筆
        assert len(comments) == 1
        assert comments[0]["id"] == parent_id
        # replies 有 1 筆
        assert len(comments[0]["replies"]) == 1
        assert comments[0]["replies"][0]["body_markdown"] == "reply comment"

    def test_nested_reply_rejected(self, db_engine):
        """reply to reply 應回 422"""
        client = _make_client(db_engine)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        parent_resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "parent"},
        )
        parent_id = parent_resp.json()["id"]

        reply_resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "reply", "parent_id": parent_id},
        )
        reply_id = reply_resp.json()["id"]

        # 嘗試 reply to reply
        nested_resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "nested reply", "parent_id": reply_id},
        )
        assert nested_resp.status_code == 422


class TestCommentPermissions:
    """Author 驗證：非 author 不能編輯/刪除"""

    def test_non_author_cannot_patch(self, db_engine):
        """非 author 的一般使用者不可 PATCH"""
        author_user = {
            "sub": "author-user",
            "email": "author@test.com",
            "preferred_username": "author@test.com",
            "modelhub_role": "reviewer",
            "company": "HurricaneEdge",
        }
        author_client = _make_client(db_engine, author_user)
        req_no = _create_submission(author_client, _SUBMISSION_EDGE)

        cr = author_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "authored comment"},
        )
        comment_id = cr.json()["id"]

        other_user = {
            "sub": "other-user",
            "email": "other@test.com",
            "preferred_username": "other@test.com",
            "modelhub_role": "reviewer",
            "company": "HurricaneEdge",
        }
        other_client = _make_client(db_engine, other_user)
        resp = other_client.patch(
            f"/api/comments/{comment_id}",
            json={"body_markdown": "hacked content"},
        )
        assert resp.status_code == 403

    def test_superadmin_can_delete_any(self, db_engine):
        """superadmin 可以刪任意人的留言"""
        author_user = {
            "sub": "author-user",
            "email": "author@test.com",
            "preferred_username": "author@test.com",
            "modelhub_role": "reviewer",
            "company": "HurricaneEdge",
        }
        author_client = _make_client(db_engine, author_user)
        req_no = _create_submission(author_client, _SUBMISSION_EDGE)

        cr = author_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "to be admin-deleted"},
        )
        comment_id = cr.json()["id"]

        admin_user = {
            "sub": "admin-user",
            "email": "admin@test.com",
            "preferred_username": "admin@test.com",
            "modelhub_role": "superadmin",
            "company": "HurricaneEdge",
        }
        admin_client = _make_client(db_engine, admin_user)
        resp = admin_client.delete(f"/api/comments/{comment_id}")
        assert resp.status_code == 204
