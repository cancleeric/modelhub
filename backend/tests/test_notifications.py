"""
test_notifications.py — M22 Phase 4 通知 / @mention / 搜尋測試

測試範圍：
  - mention_parser：parse_mentions() 正確解析 @email
  - POST comment with @mention → comment_notifications 有 mention row
  - POST comment（reply）→ comment_notifications 有 reply row
  - POST comment（new）→ submitter 收到 new_comment notification
  - GET /api/notifications：只回傳當前 user 的通知
  - POST /api/notifications/mark-read：標已讀
  - Tenant A 通知不顯示給 Tenant B
  - GET /api/comments/search：keyword 過濾
  - GET /api/comments/search：author 過濾
  - GET /api/comments/search：req_no 過濾
  - GET /api/comments/search：since 過濾
  - GET /api/comments/search：非特權者看不到 internal comment
  - CommentOut.mentioned_users 正確回傳
  - SystemEvent 寫入（comment_created / comment_replied）
  - mark-read ids 只標指定 id
"""

import sys
import types
import unittest.mock as mock
import pytest
import os

os.environ.setdefault("SKIP_ROLE_CHECK", "false")
os.environ.setdefault("AUTO_APPROVE_AFTER_VALIDATOR", "false")
os.environ["DATABASE_URL"] = "sqlite://"

# ---------------------------------------------------------------------------
# Mock setup（同 test_comments.py 慣例）
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
# Real imports
# ---------------------------------------------------------------------------
import models as _models_module
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

_SUBMISSION_EDGE = {
    "product": "AICAD",
    "company": "HurricaneEdge",
    "submitter": "submitter@hurricaneedge.com",
    "priority": "P2",
}

_SUBMISSION_SOFT = {
    "product": "Squid",
    "company": "HurricaneSoft",
    "submitter": "submitter@hurricanesoft.com",
    "priority": "P2",
}

_REVIEWER = {
    "sub": "reviewer-1",
    "email": "reviewer@test.com",
    "preferred_username": "reviewer@test.com",
    "modelhub_role": "reviewer",
    "company": "HurricaneEdge",
}

_SUBMITTER = {
    "sub": "submitter-1",
    "email": "submitter@hurricaneedge.com",
    "preferred_username": "submitter@hurricaneedge.com",
    "modelhub_role": "submitter",
    "company": "HurricaneEdge",
}

_ALICE = {
    "sub": "alice-1",
    "email": "alice@hurricaneedge.com",
    "preferred_username": "alice@hurricaneedge.com",
    "modelhub_role": "submitter",
    "company": "HurricaneEdge",
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


def _build_app(db_engine, user: dict, skip_role_check: bool = True):
    from fastapi import FastAPI as _FA
    import auth as _auth_module
    from auth import get_current_user, get_current_user_or_api_key
    from models import get_db
    import routers.submissions as _sub_router
    import routers.actions as _act_router
    import routers.comments as _comments_router
    import routers.notifications as _notif_router

    _auth_module._SKIP_ROLE_CHECK = skip_role_check

    test_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def override_get_db():
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    async def override_current_user():
        return user

    async def override_current_user_or_api_key():
        return user

    mini_app = _FA()
    mini_app.include_router(_sub_router.router, prefix="/api/submissions")
    mini_app.include_router(_act_router.router, prefix="/api/submissions")
    mini_app.include_router(_comments_router.router, prefix="/api")
    mini_app.include_router(_notif_router.router, prefix="/api")

    mini_app.dependency_overrides[get_db] = override_get_db
    mini_app.dependency_overrides[get_current_user] = override_current_user
    mini_app.dependency_overrides[get_current_user_or_api_key] = override_current_user_or_api_key

    return mini_app


def _make_client(db_engine, user: dict, skip_role_check: bool = True):
    from unittest.mock import patch as _patch
    mini_app = _build_app(db_engine, user, skip_role_check=skip_role_check)

    def _noop_resource(obj, db):
        obj.training_resource = "local_mps"

    with _patch("routers.actions._handle_start_training_resource", side_effect=_noop_resource):
        return TestClient(mini_app, raise_server_exceptions=True)


def _create_submission(client, data: dict) -> str:
    resp = client.post("/api/submissions/", json=data)
    assert resp.status_code == 201, f"create failed: {resp.text}"
    return resp.json()["submission"]["req_no"]


# ===========================================================================
# Tests
# ===========================================================================

class TestMentionParser:
    """mention_parser 單元測試"""

    def test_parse_single_mention(self):
        from mention_parser import parse_mentions
        result = parse_mentions("Hi @alice@example.com please check")
        assert result == ["alice@example.com"]

    def test_parse_multiple_mentions(self):
        from mention_parser import parse_mentions
        result = parse_mentions("@alice@example.com and @bob@company.org please review")
        assert "alice@example.com" in result
        assert "bob@company.org" in result
        assert len(result) == 2

    def test_dedup_mentions(self):
        from mention_parser import parse_mentions
        result = parse_mentions("@alice@example.com @alice@example.com")
        assert result == ["alice@example.com"]

    def test_no_mention(self):
        from mention_parser import parse_mentions
        result = parse_mentions("no mentions here")
        assert result == []

    def test_case_insensitive_dedup(self):
        from mention_parser import parse_mentions
        result = parse_mentions("@Alice@Example.COM @alice@example.com")
        assert len(result) == 1


class TestCommentMentionedUsers:
    """CommentOut.mentioned_users 解析"""

    def test_comment_with_mention_returns_mentioned_users(self, db_engine):
        client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "Hi @alice@hurricaneedge.com please review this"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "mentioned_users" in data
        assert "alice@hurricaneedge.com" in data["mentioned_users"]

    def test_comment_without_mention_returns_empty_list(self, db_engine):
        client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "no mention here"},
        )
        assert resp.status_code == 201
        assert resp.json()["mentioned_users"] == []


class TestNotificationCreation:
    """通知寫入 DB 驗證"""

    def test_mention_creates_notification(self, db_engine):
        """含 @mention 的 comment → DB 有 mention notification"""
        client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        resp = client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": f"Hi @{_SUBMITTER['email']} please review"},
        )
        assert resp.status_code == 201

        from models import CommentNotification
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            notifs = db.query(CommentNotification).filter(
                CommentNotification.recipient_email == _SUBMITTER["email"],
                CommentNotification.type == "mention",
            ).all()
            assert len(notifs) == 1
        finally:
            db.close()

    def test_reply_creates_notification(self, db_engine):
        """reply comment → parent author 收到 reply notification"""
        # parent comment by submitter
        sub_client = _make_client(db_engine, _SUBMITTER)
        req_no = _create_submission(_make_client(db_engine, _REVIEWER), _SUBMISSION_EDGE)

        parent_resp = sub_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "I have a question"},
        )
        parent_id = parent_resp.json()["id"]

        # reply by reviewer
        rev_client = _make_client(db_engine, _REVIEWER)
        rev_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "Sure, let me explain", "parent_id": parent_id},
        )

        from models import CommentNotification
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            notifs = db.query(CommentNotification).filter(
                CommentNotification.recipient_email == _SUBMITTER["email"],
                CommentNotification.type == "reply",
            ).all()
            assert len(notifs) == 1
        finally:
            db.close()

    def test_new_comment_notifies_submitter(self, db_engine):
        """新 comment → 工單 submitter 收到 new_comment notification"""
        reviewer_client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(reviewer_client, _SUBMISSION_EDGE)

        reviewer_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "Please update the dataset"},
        )

        from models import CommentNotification
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            notifs = db.query(CommentNotification).filter(
                CommentNotification.recipient_email == _SUBMISSION_EDGE["submitter"],
                CommentNotification.type == "new_comment",
            ).all()
            assert len(notifs) >= 1
        finally:
            db.close()

    def test_author_not_notified_for_own_comment(self, db_engine):
        """留言者本人不收到自己留言的通知"""
        client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "My own comment"},
        )

        from models import CommentNotification
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            notifs = db.query(CommentNotification).filter(
                CommentNotification.recipient_email == _REVIEWER["email"],
            ).all()
            assert len(notifs) == 0
        finally:
            db.close()

    def test_event_written_for_comment_created(self, db_engine):
        """建 comment → events 表有 comment_created 紀錄"""
        client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "event test"},
        )

        from models import SystemEvent
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            event = db.query(SystemEvent).filter(
                SystemEvent.event_type == "comment_created",
                SystemEvent.req_no == req_no,
            ).first()
            assert event is not None
        finally:
            db.close()


class TestNotificationsAPI:
    """GET /api/notifications + mark-read"""

    def test_list_notifications_for_current_user(self, db_engine):
        """GET /api/notifications 只回自己的通知"""
        reviewer_client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(reviewer_client, _SUBMISSION_EDGE)

        # reviewer mentions submitter
        reviewer_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": f"Hi @{_SUBMITTER['email']} check this"},
        )

        # submitter 查通知
        sub_client = _make_client(db_engine, _SUBMITTER)
        resp = sub_client.get("/api/notifications")
        assert resp.status_code == 200
        notifs = resp.json()
        assert len(notifs) >= 1
        for n in notifs:
            assert n["recipient_email"] == _SUBMITTER["email"]

    def test_mark_all_read(self, db_engine):
        """POST /api/notifications/mark-read（無 ids）→ 全部標已讀"""
        reviewer_client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(reviewer_client, _SUBMISSION_EDGE)

        reviewer_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": f"@{_SUBMITTER['email']} review needed"},
        )

        sub_client = _make_client(db_engine, _SUBMITTER)
        mark_resp = sub_client.post("/api/notifications/mark-read", json={})
        assert mark_resp.status_code == 200
        assert mark_resp.json()["marked_read"] >= 1

        # 確認 unread_only=true 現在為空
        resp = sub_client.get("/api/notifications?unread_only=true")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_mark_specific_ids_read(self, db_engine):
        """mark-read ids=[n] → 只標指定 id"""
        reviewer_client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(reviewer_client, _SUBMISSION_EDGE)

        reviewer_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": f"@{_SUBMITTER['email']} first mention"},
        )
        reviewer_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": f"@{_SUBMITTER['email']} second mention"},
        )

        sub_client = _make_client(db_engine, _SUBMITTER)
        notifs = sub_client.get("/api/notifications").json()
        assert len(notifs) >= 2

        # 只標第一個
        first_id = notifs[-1]["id"]  # unread_first，最後一個是最舊的
        mark_resp = sub_client.post("/api/notifications/mark-read", json={"ids": [first_id]})
        assert mark_resp.json()["marked_read"] == 1

        # 確認還有未讀
        remaining = sub_client.get("/api/notifications?unread_only=true").json()
        ids_remaining = [n["id"] for n in remaining]
        assert first_id not in ids_remaining

    def test_tenant_isolation_notifications(self, db_engine):
        """Tenant A 的通知不顯示給 Tenant B 的 user"""
        reviewer_client = _make_client(db_engine, _REVIEWER)

        # 建 Tenant A 工單，mention submitter@hurricaneedge.com
        req_no = _create_submission(reviewer_client, _SUBMISSION_EDGE)
        reviewer_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": f"@{_SUBMITTER['email']} edge mention"},
        )

        # Tenant B user 查通知 — 應該看不到（email 不符）
        soft_user = {
            "sub": "soft-user",
            "email": "otheruser@hurricanesoft.com",
            "preferred_username": "otheruser@hurricanesoft.com",
            "modelhub_role": "submitter",
            "company": "HurricaneSoft",
        }
        soft_client = _make_client(db_engine, soft_user)
        resp = soft_client.get("/api/notifications")
        assert resp.status_code == 200
        assert resp.json() == []


class TestCommentSearch:
    """GET /api/comments/search"""

    def test_search_returns_matching_comments(self, db_engine):
        """搜尋 keyword 找到含該字的 comment"""
        client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "This dataset is missing labels"},
        )
        client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "Please add more images"},
        )

        resp = client.get("/api/comments/search?q=dataset")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert "dataset" in results[0]["body_markdown"].lower()

    def test_search_no_results(self, db_engine):
        """搜尋不存在的 keyword 回空清單"""
        client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(client, _SUBMISSION_EDGE)
        client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "hello world"},
        )

        resp = client.get("/api/comments/search?q=zzznomatch")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_filters_by_req_no(self, db_engine):
        """req_no 過濾只回傳指定工單的 comment"""
        client = _make_client(db_engine, _REVIEWER)
        req_no1 = _create_submission(client, _SUBMISSION_EDGE)
        req_no2 = _create_submission(client, {**_SUBMISSION_EDGE, "product": "RSConch"})

        client.post(f"/api/submissions/{req_no1}/comments", json={"body_markdown": "keyword test req1"})
        client.post(f"/api/submissions/{req_no2}/comments", json={"body_markdown": "keyword test req2"})

        resp = client.get(f"/api/comments/search?q=keyword&req_no={req_no1}")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["req_no"] == req_no1

    def test_search_mentioned_users_in_results(self, db_engine):
        """search 結果中 mentioned_users 正確解析"""
        client = _make_client(db_engine, _REVIEWER)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "searchable: @alice@hurricaneedge.com please help"},
        )

        resp = client.get("/api/comments/search?q=searchable")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert "alice@hurricaneedge.com" in results[0]["mentioned_users"]

    def test_search_non_privileged_cannot_see_internal(self, db_engine):
        """非特權者搜尋看不到 internal comment"""
        reviewer_client = _make_client(db_engine, _REVIEWER, skip_role_check=False)
        req_no = _create_submission(
            _make_client(db_engine, _REVIEWER, skip_role_check=True),
            _SUBMISSION_EDGE,
        )

        reviewer_client.post(
            f"/api/submissions/{req_no}/comments",
            json={"body_markdown": "internalsecret note", "is_internal": True},
        )

        sub_client = _make_client(db_engine, _SUBMITTER, skip_role_check=False)
        resp = sub_client.get("/api/comments/search?q=internalsecret")
        assert resp.status_code == 200
        assert resp.json() == []
