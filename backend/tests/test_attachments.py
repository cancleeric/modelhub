"""
test_attachments.py — M22 Attachment API 測試

測試範圍：
  - POST /api/submissions/{req_no}/attachments（上傳、大小限制、MIME 白名單）
  - GET /api/attachments/{id}（下載，tenant 隔離）
  - DELETE /api/attachments/{id}（物理刪除，uploader 或 superadmin）
  - 上傳 5MB PNG → success
  - 上傳 11MB → 422
  - 上傳 .exe (application/octet-stream) → 422
  - tenant A 無法下載 tenant B 附件
"""

import io
import os
import sys
import tempfile
import types
import unittest.mock as mock
import pytest

os.environ.setdefault("SKIP_ROLE_CHECK", "false")
os.environ.setdefault("AUTO_APPROVE_AFTER_VALIDATORS", "false")
os.environ["DATABASE_URL"] = "sqlite://"

# ---------------------------------------------------------------------------
# Mock setup
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
import models as _models_module
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

_SUBMISSION_EDGE = {
    "product": "AICAD",
    "company": "HurricaneEdge",
    "submitter": "edge-user@edge.com",
    "priority": "P2",
}

_SUBMISSION_SOFT = {
    "product": "Squid",
    "company": "HurricaneSoft",
    "submitter": "soft-user@soft.com",
    "priority": "P2",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def tmp_attachment_dir(tmp_path):
    """臨時目錄作為附件儲存根目錄"""
    att_dir = tmp_path / "attachments"
    att_dir.mkdir()
    return str(att_dir)


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


def _build_app(db_engine, user: dict, attachment_dir: str):
    from fastapi import FastAPI as _FA
    import auth as _auth_module
    from auth import get_current_user, get_current_user_or_api_key
    from models import get_db
    import routers.submissions as _sub_router
    import routers.actions as _act_router
    import routers.attachments as _att_router

    _auth_module._SKIP_ROLE_CHECK = False
    # 覆蓋 attachment 路徑
    import routers.attachments as _att_module
    _att_module._ATTACHMENT_BASE_DIR = __import__("pathlib").Path(attachment_dir)

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
    mini_app.include_router(_att_router.router, prefix="/api")

    mini_app.dependency_overrides[get_db] = override_get_db
    mini_app.dependency_overrides[get_current_user] = override_current_user
    mini_app.dependency_overrides[get_current_user_or_api_key] = override_current_user_or_api_key

    return mini_app


def _make_client(db_engine, user: dict, attachment_dir: str):
    from unittest.mock import patch as _patch
    mini_app = _build_app(db_engine, user, attachment_dir)

    def _noop_resource(obj, db):
        obj.training_resource = "local_mps"

    with _patch("routers.actions._handle_start_training_resource", side_effect=_noop_resource):
        return TestClient(mini_app, raise_server_exceptions=True)


def _create_submission(client, data: dict) -> str:
    resp = client.post("/api/submissions/", json=data)
    assert resp.status_code == 201
    return resp.json()["submission"]["req_no"]


_REVIEWER = {
    "sub": "reviewer-user",
    "email": "reviewer@test.com",
    "preferred_username": "reviewer@test.com",
    "modelhub_role": "reviewer",
    "company": "HurricaneEdge",
}

_EDGE_SUBMITTER = {
    "sub": "edge-sub",
    "email": "edgesub@edge.com",
    "preferred_username": "edgesub@edge.com",
    "modelhub_role": "submitter",
    "company": "HurricaneEdge",
}

_SOFT_SUBMITTER = {
    "sub": "soft-sub",
    "email": "softsub@soft.com",
    "preferred_username": "softsub@soft.com",
    "modelhub_role": "submitter",
    "company": "HurricaneSoft",
}


# ===========================================================================
# Tests
# ===========================================================================

class TestAttachmentUpload:
    """上傳 API 測試"""

    def test_upload_small_pdf_success(self, db_engine, tmp_attachment_dir):
        """上傳 5MB PDF → 成功"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        content = b"%PDF-1.4 " + b"x" * (5 * 1024 * 1024)  # ~5MB
        resp = client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("test.pdf", io.BytesIO(content), "application/pdf")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "test.pdf"
        assert data["size_bytes"] == len(content)
        assert data["mime_type"] == "application/pdf"
        assert "url" in data

    def test_upload_image_success(self, db_engine, tmp_attachment_dir):
        """上傳 PNG → 成功"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        content = b"\x89PNG\r\n" + b"x" * 1024
        resp = client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("photo.png", io.BytesIO(content), "image/png")},
        )
        assert resp.status_code == 201
        assert resp.json()["mime_type"] == "image/png"

    def test_upload_text_file_success(self, db_engine, tmp_attachment_dir):
        """上傳 text/plain → 成功"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        content = b"# Data Spec\n\nsome text"
        resp = client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("spec.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 201

    def test_upload_zip_success(self, db_engine, tmp_attachment_dir):
        """上傳 application/zip → 成功"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        content = b"PK" + b"x" * 100
        resp = client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("archive.zip", io.BytesIO(content), "application/zip")},
        )
        assert resp.status_code == 201

    def test_upload_oversized_file_rejected(self, db_engine, tmp_attachment_dir):
        """上傳 > 10MB → 422"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        content = b"x" * (11 * 1024 * 1024)  # 11 MB
        resp = client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("bigfile.pdf", io.BytesIO(content), "application/pdf")},
        )
        assert resp.status_code == 422
        assert "10 MB" in resp.json()["detail"] or "10485760" in resp.json()["detail"]

    def test_upload_disallowed_mime_rejected(self, db_engine, tmp_attachment_dir):
        """上傳 application/octet-stream (.exe) → 422"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        content = b"MZ" + b"\x00" * 100  # PE header mock
        resp = client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("malware.exe", io.BytesIO(content), "application/octet-stream")},
        )
        assert resp.status_code == 422
        assert "not allowed" in resp.json()["detail"]

    def test_upload_stores_file_on_disk(self, db_engine, tmp_attachment_dir):
        """上傳後檔案確實存在於 storage_path"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        content = b"hello world pdf"
        resp = client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("check.pdf", io.BytesIO(content), "application/pdf")},
        )
        assert resp.status_code == 201

        att_id = resp.json()["id"]
        from models import SubmissionAttachment
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            att = db.query(SubmissionAttachment).filter(SubmissionAttachment.id == att_id).first()
            assert att is not None
            assert os.path.exists(att.storage_path)
            with open(att.storage_path, "rb") as f:
                assert f.read() == content
        finally:
            db.close()


class TestAttachmentDownload:
    """下載 API 測試"""

    def test_download_returns_correct_bytes(self, db_engine, tmp_attachment_dir):
        """下載回傳原始 bytes"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        content = b"original pdf content"
        upload_resp = client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("orig.pdf", io.BytesIO(content), "application/pdf")},
        )
        att_id = upload_resp.json()["id"]

        download_resp = client.get(f"/api/attachments/{att_id}")
        assert download_resp.status_code == 200
        assert download_resp.content == content

    def test_download_404_for_missing(self, db_engine, tmp_attachment_dir):
        """不存在的 attachment id → 404"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        resp = client.get("/api/attachments/99999")
        assert resp.status_code == 404

    def test_tenant_isolation_download(self, db_engine, tmp_attachment_dir):
        """tenant A 無法下載 tenant B 附件"""
        # Reviewer 上傳 HurricaneSoft 工單附件
        reviewer_client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        soft_req_no = _create_submission(reviewer_client, _SUBMISSION_SOFT)
        content = b"soft secret data"
        upload_resp = reviewer_client.post(
            f"/api/submissions/{soft_req_no}/attachments",
            files={"file": ("secret.pdf", io.BytesIO(content), "application/pdf")},
        )
        att_id = upload_resp.json()["id"]

        # HurricaneEdge submitter 嘗試下載
        edge_client = _make_client(db_engine, _EDGE_SUBMITTER, tmp_attachment_dir)
        resp = edge_client.get(f"/api/attachments/{att_id}")
        assert resp.status_code == 403


class TestAttachmentDelete:
    """刪除 API 測試"""

    def test_uploader_can_delete(self, db_engine, tmp_attachment_dir):
        """uploader 可以刪自己的附件"""
        client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(client, _SUBMISSION_EDGE)

        content = b"to be deleted"
        upload_resp = client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("del.pdf", io.BytesIO(content), "application/pdf")},
        )
        att_id = upload_resp.json()["id"]

        # 確認檔案存在
        from models import SubmissionAttachment
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            att = db.query(SubmissionAttachment).filter(SubmissionAttachment.id == att_id).first()
            storage_path = att.storage_path
        finally:
            db.close()

        del_resp = client.delete(f"/api/attachments/{att_id}")
        assert del_resp.status_code == 204
        assert not os.path.exists(storage_path)

    def test_non_uploader_cannot_delete(self, db_engine, tmp_attachment_dir):
        """非 uploader 的一般使用者不可刪"""
        uploader_client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(uploader_client, _SUBMISSION_EDGE)

        content = b"protected"
        upload_resp = uploader_client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("prot.pdf", io.BytesIO(content), "application/pdf")},
        )
        att_id = upload_resp.json()["id"]

        other_user = {
            "sub": "other",
            "email": "other@test.com",
            "preferred_username": "other@test.com",
            "modelhub_role": "reviewer",
            "company": "HurricaneEdge",
        }
        other_client = _make_client(db_engine, other_user, tmp_attachment_dir)
        resp = other_client.delete(f"/api/attachments/{att_id}")
        assert resp.status_code == 403

    def test_superadmin_can_delete_any(self, db_engine, tmp_attachment_dir):
        """superadmin 可刪任意人的附件"""
        uploader_client = _make_client(db_engine, _REVIEWER, tmp_attachment_dir)
        req_no = _create_submission(uploader_client, _SUBMISSION_EDGE)

        content = b"admin target"
        upload_resp = uploader_client.post(
            f"/api/submissions/{req_no}/attachments",
            files={"file": ("admin_del.pdf", io.BytesIO(content), "application/pdf")},
        )
        att_id = upload_resp.json()["id"]

        admin_user = {
            "sub": "admin",
            "email": "admin@test.com",
            "preferred_username": "admin@test.com",
            "modelhub_role": "superadmin",
            "company": "HurricaneEdge",
        }
        admin_client = _make_client(db_engine, admin_user, tmp_attachment_dir)
        resp = admin_client.delete(f"/api/attachments/{att_id}")
        assert resp.status_code == 204
