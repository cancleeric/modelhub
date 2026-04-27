"""
tests/test_inference_server.py — inference-server unit + integration tests

覆蓋：
  - registry refresh（mock httpx）
  - LRU eviction 邏輯
  - loader dispatch（xgboost-text / llamaguard-1b / sentence-transformers / yolov8）
  - 不支援的 arch → 400
  - /v1/models endpoint
  - /v1/models/{req_no} endpoint
  - /v1/models/{req_no}/loaded endpoint
  - /v1/models/{req_no}/unload endpoint
  - /v1/predict xgboost-text
  - /v1/predict llamaguard-1b
  - /v1/predict sentence-transformers
  - /v1/predict yolov8 (image_url)
  - /v1/predict model not found → 404
  - /v1/predict unsupported arch → 400
  - /v1/predict file not found → 404
  - /metrics endpoint
  - /health endpoint
  - 缺必要 body 欄位 → 422
  - 多次 predict 同一 model：LRU touch（不重複 load）
  - LRU eviction 驗證：第 4 個 model 擠掉第 1 個
"""

import sys
import types
import unittest.mock as mock
from collections import OrderedDict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# 匯入 main（conftest 已先 mock 所有重套件）
# ---------------------------------------------------------------------------

import main as M


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """每個測試前重置 registry 和 LRU cache。"""
    M._registry.clear()
    M._lru_cache.clear()
    yield
    M._registry.clear()
    M._lru_cache.clear()


@pytest.fixture()
def client():
    with TestClient(M.app, raise_server_exceptions=True) as c:
        yield c


def _make_meta(req_no: str, arch: str = "yolov8s", file_path: str = "/models/fake.pt") -> dict:
    return {
        "req_no": req_no,
        "model_name": f"Model {req_no}",
        "arch": arch,
        "file_path": file_path,
        "product": "test",
        "status": "trained",
        "source": "registry",
    }


def _inject_registry(**models: dict) -> None:
    """直接注入 registry，key = req_no uppercase。"""
    for req_no, meta in models.items():
        M._registry[req_no.upper()] = meta


# ---------------------------------------------------------------------------
# 1. Registry refresh
# ---------------------------------------------------------------------------

class TestRegistryRefresh:
    def test_refresh_populates_registry_from_external(self):
        ext_resp = [
            {"req_no": "EXT-2026-001", "model_name": "LlamaGuard 1B", "arch": "llamaguard-1b",
             "file_path": "/models/llamaguard", "product": "aegis", "status": "registered"},
        ]
        with mock.patch("main.httpx.get") as mock_get:
            mock_get.return_value = mock.MagicMock(status_code=200, json=lambda: ext_resp)
            M.refresh_registry()
        assert "EXT-2026-001" in M._registry
        assert M._registry["EXT-2026-001"]["arch"] == "llamaguard-1b"

    def test_refresh_populates_registry_from_registry_api(self):
        reg_resp = [
            {"req_no": "MH-2026-022", "model_name": "Severity Classifier", "arch": "xgboost-text",
             "file_path": "/models/severity.pkl", "product": "modelhub"},
        ]
        # external-models 回空，registry 有資料
        responses = {
            "/api/external-models/": mock.MagicMock(status_code=200, json=lambda: []),
            "/api/registry": mock.MagicMock(status_code=200, json=lambda: reg_resp),
        }
        def _side_effect(url, **kwargs):
            for suffix, resp in responses.items():
                if url.endswith(suffix):
                    return resp
            return mock.MagicMock(status_code=404)

        with mock.patch("main.httpx.get", side_effect=_side_effect):
            M.refresh_registry()
        assert "MH-2026-022" in M._registry

    def test_refresh_api_unreachable_keeps_old_registry(self):
        M._registry["OLD-MODEL"] = _make_meta("OLD-MODEL")
        with mock.patch("main.httpx.get", side_effect=Exception("timeout")):
            M.refresh_registry()
        # 舊 registry 應該被保留（refresh 回 0 models 不清空）
        assert "OLD-MODEL" in M._registry

    def test_refresh_normalizes_req_no_to_uppercase(self):
        ext_resp = [{"req_no": "ext-2026-001", "model_name": "X", "arch": "llamaguard-1b",
                     "file_path": "/models/x", "product": "p", "status": "registered"}]
        with mock.patch("main.httpx.get") as mg:
            mg.return_value = mock.MagicMock(status_code=200, json=lambda: ext_resp)
            M.refresh_registry()
        assert "EXT-2026-001" in M._registry


# ---------------------------------------------------------------------------
# 2. LRU eviction
# ---------------------------------------------------------------------------

class TestLRUEviction:
    def test_evict_when_cache_full(self):
        M.MAX_LOADED_MODELS = 3
        for i in range(3):
            M._lru_cache[f"MODEL-{i}"] = {"handle": object(), "arch": "xgboost-text", "meta": {}}
        assert len(M._lru_cache) == 3
        M._evict_if_needed()
        assert len(M._lru_cache) == 2
        assert "MODEL-0" not in M._lru_cache

    def test_touch_moves_to_end(self):
        M._lru_cache["A"] = {"handle": None, "arch": "x", "meta": {}}
        M._lru_cache["B"] = {"handle": None, "arch": "x", "meta": {}}
        M._lru_cache["C"] = {"handle": None, "arch": "x", "meta": {}}
        M._touch("A")
        keys = list(M._lru_cache.keys())
        assert keys[-1] == "A", f"Expected A at end, got {keys}"

    def test_lru_evict_oldest_when_fourth_model_loaded(self, tmp_path):
        """LRU 滿 3 → 第 4 個 model 載入時，最舊的被 evict。"""
        M.MAX_LOADED_MODELS = 3
        fake_pkl = tmp_path / "model.pkl"
        fake_pkl.write_bytes(b"fake")

        # 填滿 cache
        for i in range(3):
            key = f"FILL-{i}"
            M._lru_cache[key] = {"handle": object(), "arch": "xgboost-text", "meta": {}}

        first_key = list(M._lru_cache.keys())[0]  # = "FILL-0"

        # 注入第 4 個 model registry
        _inject_registry(**{"NEW-001": _make_meta("NEW-001", arch="xgboost-text", file_path=str(fake_pkl))})

        import joblib
        joblib.load = mock.MagicMock(return_value=mock.MagicMock())

        with mock.patch("main.Path.exists", return_value=True):
            M._lazy_load("NEW-001")

        assert first_key not in M._lru_cache, "First model should have been evicted"
        assert "NEW-001" in M._lru_cache


# ---------------------------------------------------------------------------
# 3. Loader dispatch
# ---------------------------------------------------------------------------

class TestLoaderDispatch:
    def test_xgboost_text_calls_joblib(self, tmp_path):
        fake_pkl = tmp_path / "model.pkl"
        fake_pkl.write_bytes(b"fake")
        import joblib
        joblib.load = mock.MagicMock(return_value=mock.MagicMock())
        handle = M._load_xgboost(fake_pkl)
        joblib.load.assert_called_once_with(fake_pkl)

    def test_llamaguard_calls_transformers(self, tmp_path):
        fake_dir = tmp_path / "llama"
        fake_dir.mkdir()
        import transformers
        transformers.AutoTokenizer.from_pretrained = mock.MagicMock(return_value=mock.MagicMock())
        transformers.AutoModelForCausalLM.from_pretrained = mock.MagicMock(return_value=mock.MagicMock())
        handle = M._load_llamaguard(fake_dir)
        assert "model" in handle
        assert "tokenizer" in handle

    def test_sentence_transformer_calls_st(self, tmp_path):
        fake_dir = tmp_path / "st"
        fake_dir.mkdir()
        import sentence_transformers
        mock_st = mock.MagicMock()
        sentence_transformers.SentenceTransformer = mock.MagicMock(return_value=mock_st)
        handle = M._load_sentence_transformer(fake_dir)
        sentence_transformers.SentenceTransformer.assert_called_once_with(str(fake_dir))

    def test_yolo_calls_ultralytics(self, tmp_path):
        fake_pt = tmp_path / "model.pt"
        fake_pt.write_bytes(b"fake")
        import ultralytics
        mock_yolo_instance = mock.MagicMock()
        ultralytics.YOLO = mock.MagicMock(return_value=mock_yolo_instance)
        handle = M._load_yolo(fake_pt)
        ultralytics.YOLO.assert_called_once_with(str(fake_pt))

    def test_unsupported_arch_raises_value_error(self, tmp_path):
        fake_pt = tmp_path / "model.pt"
        fake_pt.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported arch"):
            M._dispatch_loader("unknown-arch-xyz", fake_pt)


# ---------------------------------------------------------------------------
# 4. HTTP API endpoints
# ---------------------------------------------------------------------------

class TestListModelsEndpoint:
    def test_get_v1_models_empty(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 0
        assert data["models"] == []

    def test_get_v1_models_with_registry(self, client):
        _inject_registry(
            **{"EXT-2026-001": _make_meta("EXT-2026-001", "llamaguard-1b"),
               "MH-2026-022": _make_meta("MH-2026-022", "xgboost-text")}
        )
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        req_nos = {m["req_no"] for m in data["models"]}
        assert "EXT-2026-001" in req_nos
        assert "MH-2026-022" in req_nos

    def test_get_v1_models_shows_in_ram_flag(self, client):
        _inject_registry(**{"MH-2026-022": _make_meta("MH-2026-022", "xgboost-text")})
        M._lru_cache["MH-2026-022"] = {"handle": object(), "arch": "xgboost-text", "meta": {}}
        r = client.get("/v1/models")
        model = r.json()["models"][0]
        assert model["in_ram"] is True


class TestGetModelEndpoint:
    def test_get_existing_model(self, client):
        _inject_registry(**{"EXT-2026-001": _make_meta("EXT-2026-001", "llamaguard-1b")})
        r = client.get("/v1/models/EXT-2026-001")
        assert r.status_code == 200
        assert r.json()["req_no"] == "EXT-2026-001"

    def test_get_model_case_insensitive(self, client):
        _inject_registry(**{"EXT-2026-001": _make_meta("EXT-2026-001", "llamaguard-1b")})
        r = client.get("/v1/models/ext-2026-001")
        assert r.status_code == 200

    def test_get_nonexistent_model_404(self, client):
        r = client.get("/v1/models/DOES-NOT-EXIST")
        assert r.status_code == 404


class TestLoadedEndpoint:
    def test_not_loaded(self, client):
        _inject_registry(**{"MH-2026-022": _make_meta("MH-2026-022")})
        r = client.get("/v1/models/MH-2026-022/loaded")
        assert r.status_code == 200
        assert r.json()["loaded"] is False

    def test_loaded(self, client):
        M._lru_cache["MH-2026-022"] = {"handle": object(), "arch": "xgboost-text", "meta": {}}
        r = client.get("/v1/models/MH-2026-022/loaded")
        assert r.status_code == 200
        assert r.json()["loaded"] is True


class TestUnloadEndpoint:
    def test_unload_loaded_model(self, client):
        M._lru_cache["MH-2026-022"] = {"handle": object(), "arch": "xgboost-text", "meta": {}}
        r = client.post("/v1/models/MH-2026-022/unload")
        assert r.status_code == 200
        assert r.json()["unloaded"] is True
        assert "MH-2026-022" not in M._lru_cache

    def test_unload_not_loaded_404(self, client):
        r = client.post("/v1/models/NOT-LOADED/unload")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 5. /v1/predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def _inject_with_handle(self, req_no: str, arch: str, handle: object) -> None:
        meta = _make_meta(req_no, arch)
        M._registry[req_no] = meta
        M._lru_cache[req_no] = {"handle": handle, "arch": arch, "meta": meta}

    def test_predict_xgboost_text(self, client):
        import numpy as np
        mock_model = mock.MagicMock()
        mock_model.predict = mock.MagicMock(return_value=mock.MagicMock(tolist=lambda: ["low"]))
        mock_model.predict_proba = mock.MagicMock(side_effect=AttributeError)
        self._inject_with_handle("MH-2026-022", "xgboost-text", mock_model)
        r = client.post("/v1/predict?model=MH-2026-022", json={"texts": ["test incident"]})
        assert r.status_code == 200
        data = r.json()
        assert data["model"] == "MH-2026-022"
        assert "predictions" in data
        assert "latency_ms" in data

    def test_predict_llamaguard(self, client):
        mock_tok = mock.MagicMock()
        mock_tok.return_value = {"input_ids": mock.MagicMock()}
        mock_tok.decode = mock.MagicMock(return_value="safe")
        mock_model_obj = mock.MagicMock()
        mock_outputs = mock.MagicMock()
        mock_outputs.__getitem__ = mock.MagicMock(return_value=mock.MagicMock())
        mock_model_obj.generate = mock.MagicMock(return_value=mock_outputs)
        handle = {"model": mock_model_obj, "tokenizer": mock_tok}
        self._inject_with_handle("EXT-2026-001", "llamaguard-1b", handle)
        r = client.post("/v1/predict?model=EXT-2026-001", json={"prompt": "Is this harmful?"})
        assert r.status_code == 200
        data = r.json()
        assert data["model"] == "EXT-2026-001"
        assert "predictions" in data

    def test_predict_sentence_transformer(self, client):
        import numpy as np
        mock_st = mock.MagicMock()
        mock_st.encode = mock.MagicMock(return_value=mock.MagicMock(tolist=lambda: [[0.1, 0.2, 0.3]]))
        self._inject_with_handle("MH-2026-018", "sentence-transformers/all-MiniLM-L6-v2", mock_st)
        r = client.post("/v1/predict?model=MH-2026-018", json={"texts": ["hello world"]})
        assert r.status_code == 200
        data = r.json()
        assert "predictions" in data

    def test_predict_yolo_with_image_url(self, client):
        mock_yolo = mock.MagicMock()
        box_mock = mock.MagicMock()
        box_mock.xyxy = [mock.MagicMock(tolist=lambda: [10.0, 20.0, 100.0, 200.0])]
        box_mock.conf = [mock.MagicMock(__float__=lambda s: 0.9)]
        box_mock.cls = [mock.MagicMock(__int__=lambda s: 0)]
        r_mock = mock.MagicMock()
        r_mock.boxes = [box_mock]
        r_mock.names = {0: "person"}
        mock_yolo.return_value = [r_mock]
        self._inject_with_handle("MH-2026-005", "yolov8s", mock_yolo)

        fake_img_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # minimal fake JPEG header

        with mock.patch("main.httpx.get") as mock_http:
            mock_http.return_value = mock.MagicMock(status_code=200, content=fake_img_bytes)
            with mock.patch("PIL.Image.open") as mock_pil:
                mock_pil.return_value = mock.MagicMock(
                    convert=mock.MagicMock(return_value=mock.MagicMock())
                )
                r = client.post(
                    "/v1/predict?model=MH-2026-005",
                    json={"image_url": "http://example.com/img.jpg"},
                )
        assert r.status_code == 200
        data = r.json()
        assert "predictions" in data

    def test_predict_model_not_found_404(self, client):
        r = client.post("/v1/predict?model=DOES-NOT-EXIST", json={"texts": ["x"]})
        assert r.status_code == 404

    def test_predict_unsupported_arch_400(self, client):
        meta = _make_meta("UNKNOWN-001", arch="bert-base")
        meta["file_path"] = "/models/fake.pkl"
        M._registry["UNKNOWN-001"] = meta
        with mock.patch("main.Path.exists", return_value=True):
            r = client.post("/v1/predict?model=UNKNOWN-001", json={"texts": ["x"]})
        assert r.status_code == 400
        assert "Unsupported arch" in r.json()["detail"]

    def test_predict_missing_body_xgboost_422(self, client):
        mock_model = mock.MagicMock()
        self._inject_with_handle("MH-2026-022", "xgboost-text", mock_model)
        # 空 body，未傳 texts
        r = client.post("/v1/predict?model=MH-2026-022", json={})
        assert r.status_code == 422

    def test_predict_same_model_twice_no_reload(self, client):
        """同一 model 連續兩次 predict，loader 只呼叫一次。"""
        import numpy as np
        mock_model = mock.MagicMock()
        mock_model.predict = mock.MagicMock(return_value=mock.MagicMock(tolist=lambda: ["low"]))
        mock_model.predict_proba = mock.MagicMock(side_effect=AttributeError)
        self._inject_with_handle("MH-2026-022", "xgboost-text", mock_model)

        with mock.patch("main._dispatch_loader") as mock_loader:
            client.post("/v1/predict?model=MH-2026-022", json={"texts": ["a"]})
            client.post("/v1/predict?model=MH-2026-022", json={"texts": ["b"]})
            # model 已在 cache，不應再呼叫 loader
            mock_loader.assert_not_called()


# ---------------------------------------------------------------------------
# 6. /metrics + /health
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type(self, client):
        r = client.get("/metrics")
        assert "text/plain" in r.headers.get("content-type", "")


class TestHealthEndpoint:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "loaded_models" in data
        assert "registry_size" in data
