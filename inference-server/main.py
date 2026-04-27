"""
ModelHub Inference Server — multi-model lazy load + LRU + unified HTTP API + prometheus

Phase 1:
  B: HTTP inference  (POST /v1/predict?model=<req_no>)
  C: registry path   (啟動時 + 每 5min 從 modelhub-api 拉 model 清單)

支援 loader dispatch：
  xgboost-text          → joblib.load
  llamaguard-1b         → transformers AutoModelForCausalLM
  sentence-transformers → sentence_transformers.SentenceTransformer
  yolov8*               → ultralytics YOLO
  其他                  → 400 + 建議

Port: 8951
"""

import logging
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from PIL import Image as PILImage
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("inference_server")

MODELHUB_API_URL: str = os.environ.get(
    "MODELHUB_API_URL", "http://modelhub-api-dev:8000"
)
MODELHUB_API_KEY: str = os.environ.get("MODELHUB_API_KEY", "")
MODELS_MOUNT: Path = Path(os.environ.get("MODELS_MOUNT", "/models"))
MAX_LOADED_MODELS: int = int(os.environ.get("MAX_LOADED_MODELS", "3"))
REGISTRY_REFRESH_SECONDS: int = int(os.environ.get("REGISTRY_REFRESH_SECONDS", "300"))

# ---------------------------------------------------------------------------
# Prometheus metrics (lazy — 只在模組載入時初始化 counter/histogram/gauge)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True
    _request_counter = Counter(
        "modelhub_inference_request_total",
        "Inference request count",
        ["model", "status"],
    )
    _duration_histogram = Histogram(
        "modelhub_inference_duration_seconds",
        "Inference latency",
        ["model"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _loaded_gauge = Gauge(
        "modelhub_inference_loaded_models",
        "Number of models currently in RAM",
    )
except ImportError:
    _PROM_AVAILABLE = False
    logger.warning("prometheus_client not installed, /metrics will return 501")


def _prom_count(model: str, status: str) -> None:
    if _PROM_AVAILABLE:
        _request_counter.labels(model=model, status=status).inc()


def _prom_observe(model: str, duration_s: float) -> None:
    if _PROM_AVAILABLE:
        _duration_histogram.labels(model=model).observe(duration_s)


def _prom_set_loaded(n: int) -> None:
    if _PROM_AVAILABLE:
        _loaded_gauge.set(n)


# ---------------------------------------------------------------------------
# Registry cache — 從 modelhub-api 拉取的 model metadata
# key: req_no (uppercase, e.g. "EXT-2026-001")
# ---------------------------------------------------------------------------

# {req_no: {req_no, model_name, arch, file_path, product, status, source}}
_registry: dict[str, dict] = {}


def _api_headers() -> dict:
    h: dict = {}
    if MODELHUB_API_KEY:
        h["X-Api-Key"] = MODELHUB_API_KEY
    return h


def _fetch_external_models() -> list[dict]:
    """GET /api/external-models/ — 列出 EXT- 已 registered model。"""
    try:
        resp = httpx.get(
            f"{MODELHUB_API_URL}/api/external-models/",
            headers=_api_headers(),
            timeout=8.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Cannot fetch external models: %s", e)
    return []


def _fetch_trained_submissions() -> list[dict]:
    """GET /api/submissions?status=trained — 列出 MH- 已訓練 model。"""
    try:
        resp = httpx.get(
            f"{MODELHUB_API_URL}/api/submissions",
            params={"status": "trained"},
            headers=_api_headers(),
            timeout=8.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            # submissions API 回傳 {"items": [...]} 或直接 list
            if isinstance(data, list):
                return data
            return data.get("items", data.get("submissions", []))
    except Exception as e:
        logger.warning("Cannot fetch trained submissions: %s", e)
    return []


def _fetch_registry_models() -> list[dict]:
    """GET /api/registry?is_current=true — 取得已驗收 model versions。"""
    try:
        resp = httpx.get(
            f"{MODELHUB_API_URL}/api/registry",
            params={"is_current": "true"},
            headers=_api_headers(),
            timeout=8.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Cannot fetch registry: %s", e)
    return []


def refresh_registry() -> None:
    """從 modelhub-api 拉取最新 model 清單，合併進 _registry。"""
    global _registry
    new_registry: dict[str, dict] = {}

    # 1. EXT- models（external_models API）
    for entry in _fetch_external_models():
        req_no = (entry.get("req_no") or "").upper().strip()
        if not req_no:
            continue
        arch = (entry.get("arch") or "").lower()
        new_registry[req_no] = {
            "req_no": req_no,
            "model_name": entry.get("model_name", req_no),
            "arch": arch,
            "file_path": entry.get("file_path"),
            "product": entry.get("product", ""),
            "status": entry.get("status", "registered"),
            "source": "external",
        }

    # 2. MH- models（registry API — 已驗收 versions）
    for entry in _fetch_registry_models():
        req_no = (entry.get("req_no") or "").upper().strip()
        if not req_no or req_no in new_registry:
            continue
        arch = (entry.get("arch") or "").lower()
        file_path = entry.get("file_path") or entry.get("model_output_path")
        new_registry[req_no] = {
            "req_no": req_no,
            "model_name": entry.get("model_name", req_no),
            "arch": arch,
            "file_path": file_path,
            "product": entry.get("product", ""),
            "status": "trained",
            "source": "registry",
        }

    if new_registry:
        _registry = new_registry
        logger.info(
            "Registry refreshed: %d models — %s",
            len(new_registry),
            list(new_registry.keys()),
        )
    else:
        logger.warning("Registry refresh returned 0 models (API may be unreachable)")


# ---------------------------------------------------------------------------
# LRU model cache — 最多 MAX_LOADED_MODELS 個 model 在 RAM
# key: req_no (uppercase)
# value: {"handle": <loaded model object>, "arch": str, "meta": dict}
# ---------------------------------------------------------------------------

_lru_cache: OrderedDict = OrderedDict()


def _evict_if_needed() -> None:
    """若超過上限，逐一 evict 最舊的 model。"""
    while len(_lru_cache) >= MAX_LOADED_MODELS:
        evicted_key, _ = _lru_cache.popitem(last=False)
        logger.info("LRU evict: %s (cache full, limit=%d)", evicted_key, MAX_LOADED_MODELS)
    _prom_set_loaded(len(_lru_cache))


def _touch(req_no: str) -> None:
    """將 req_no 移到 OrderedDict 尾端（最近使用）。"""
    if req_no in _lru_cache:
        _lru_cache.move_to_end(req_no)


def _resolve_file_path(meta: dict) -> Optional[Path]:
    """
    解析 model 的 file_path。
    先查 MODELS_MOUNT/<relative>，fallback 絕對路徑。
    """
    raw = meta.get("file_path")
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    # 相對路徑：嘗試在 MODELS_MOUNT 下找
    candidate = MODELS_MOUNT / p
    if candidate.exists():
        return candidate
    # 直接用絕對路徑（即使 bind mount 路徑不通，留給呼叫方判斷）
    return p if p.is_absolute() else None


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------

_SUPPORTED_ARCH_PREFIXES = (
    "xgboost-text",
    "llamaguard-1b",
    "sentence-transformers",
    "yolov8",
)


def _dispatch_loader(arch: str, file_path: Path) -> object:
    """
    根據 arch 選擇 loader，回傳 loaded model handle。
    不支援的 arch 拋 ValueError。
    """
    arch_lower = arch.lower()

    if arch_lower == "xgboost-text":
        return _load_xgboost(file_path)
    elif arch_lower == "llamaguard-1b":
        return _load_llamaguard(file_path)
    elif arch_lower.startswith("sentence-transformers"):
        return _load_sentence_transformer(file_path)
    elif arch_lower.startswith("yolov8"):
        return _load_yolo(file_path)
    else:
        raise ValueError(
            f"Unsupported arch '{arch}'. Supported prefixes: {list(_SUPPORTED_ARCH_PREFIXES)}. "
            "Please add a loader in inference-server/main.py and open a PR."
        )


def _load_xgboost(file_path: Path) -> object:
    try:
        import joblib  # type: ignore
    except ImportError:
        raise RuntimeError("joblib not installed; run: pip install joblib")
    logger.info("Loading xgboost-text model from %s", file_path)
    return joblib.load(file_path)


def _load_llamaguard(file_path: Path) -> object:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        raise RuntimeError("transformers not installed")
    logger.info("Loading llamaguard-1b from %s", file_path)
    tokenizer = AutoTokenizer.from_pretrained(str(file_path))
    model = AutoModelForCausalLM.from_pretrained(str(file_path))
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


def _load_sentence_transformer(file_path: Path) -> object:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise RuntimeError("sentence-transformers not installed")
    logger.info("Loading sentence-transformer from %s", file_path)
    return SentenceTransformer(str(file_path))


def _load_yolo(file_path: Path) -> object:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        raise RuntimeError("ultralytics not installed")
    logger.info("Loading YOLO from %s", file_path)
    return YOLO(str(file_path))


def _lazy_load(req_no: str) -> dict:
    """
    若已在 LRU cache 中直接回傳（touch）；否則載入並加入 cache。
    回傳 cache entry dict。
    """
    req_no = req_no.upper()

    if req_no in _lru_cache:
        _touch(req_no)
        return _lru_cache[req_no]

    # 查 registry
    meta = _registry.get(req_no)
    if meta is None:
        # 嘗試 refresh 一次再查
        try:
            refresh_registry()
        except Exception:
            pass
        meta = _registry.get(req_no)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Model {req_no} not found in registry")

    arch = meta.get("arch", "")
    if not arch:
        raise HTTPException(
            status_code=400,
            detail=f"Model {req_no} has no arch set in registry; cannot determine loader",
        )

    file_path = _resolve_file_path(meta)
    if file_path is None or not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found for {req_no} (path={meta.get('file_path')}). "
                   "Check bind mount /models.",
        )

    # Check arch support before eviction
    arch_lower = arch.lower()
    if not any(arch_lower.startswith(p) for p in _SUPPORTED_ARCH_PREFIXES):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported arch '{arch}' for {req_no}. "
                   f"Supported prefixes: {list(_SUPPORTED_ARCH_PREFIXES)}. "
                   "Please add a loader and open a PR.",
        )

    # Evict if needed, then load
    _evict_if_needed()
    try:
        handle = _dispatch_loader(arch, file_path)
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to load %s: %s", req_no, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    entry = {"handle": handle, "arch": arch, "meta": meta}
    _lru_cache[req_no] = entry
    _prom_set_loaded(len(_lru_cache))
    logger.info("Loaded %s (arch=%s), cache size=%d/%d", req_no, arch, len(_lru_cache), MAX_LOADED_MODELS)
    return entry


# ---------------------------------------------------------------------------
# Predict helpers — arch-specific inference
# ---------------------------------------------------------------------------

def _predict_xgboost(handle: object, body: dict) -> dict:
    texts: list = body.get("texts", [])
    if not texts or not isinstance(texts, list):
        raise HTTPException(status_code=422, detail="xgboost-text requires body: {texts: [...]}")
    import numpy as np  # type: ignore
    predictions = handle.predict(texts)  # type: ignore
    scores = None
    try:
        proba = handle.predict_proba(texts)  # type: ignore
        scores = proba.tolist()
    except Exception:
        pass
    result: dict = {"predictions": predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)}
    if scores is not None:
        result["scores"] = scores
    return result


def _predict_llamaguard(handle: object, body: dict) -> dict:
    prompt: str = body.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=422, detail="llamaguard-1b requires body: {prompt: '...'}")
    import torch  # type: ignore
    model = handle["model"]  # type: ignore
    tokenizer = handle["tokenizer"]  # type: ignore
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # LlamaGuard 回傳 "safe" 或 "unsafe\n<category>"
    lines = generated.strip().splitlines()
    verdict = lines[0].strip() if lines else generated.strip()
    categories = [l.strip() for l in lines[1:] if l.strip()] if len(lines) > 1 else []
    return {
        "predictions": [{"verdict": verdict, "categories": categories}],
        "raw": generated,
    }


def _predict_sentence_transformer(handle: object, body: dict) -> dict:
    texts: list = body.get("texts", [])
    if not texts or not isinstance(texts, list):
        raise HTTPException(
            status_code=422,
            detail="sentence-transformers requires body: {texts: [...]}"
        )
    embeddings = handle.encode(texts, show_progress_bar=False)  # type: ignore
    return {"predictions": embeddings.tolist()}


def _predict_yolo(handle: object, body: dict) -> dict:
    import io as _io
    import httpx as _httpx

    image_url: Optional[str] = body.get("image_url")
    image_b64: Optional[str] = body.get("image_b64")

    if image_b64:
        import base64
        raw = image_b64
        if "," in raw:
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)
    elif image_url:
        resp = _httpx.get(image_url, timeout=10.0, follow_redirects=True)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Cannot fetch image_url (HTTP {resp.status_code})")
        img_bytes = resp.content
    else:
        raise HTTPException(
            status_code=422,
            detail="yolov8 requires body: {image_url: '...'} or {image_b64: '...'}",
        )

    pil_img = PILImage.open(_io.BytesIO(img_bytes)).convert("RGB")
    results = handle(pil_img, verbose=False)  # type: ignore
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names.get(cls_id, str(cls_id))
            detections.append({
                "label": label,
                "class_id": cls_id,
                "confidence": round(conf, 6),
                "box": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                },
            })
    return {"predictions": detections, "count": len(detections)}


def _run_predict(arch: str, handle: object, body: dict) -> dict:
    arch_lower = arch.lower()
    if arch_lower == "xgboost-text":
        return _predict_xgboost(handle, body)
    elif arch_lower == "llamaguard-1b":
        return _predict_llamaguard(handle, body)
    elif arch_lower.startswith("sentence-transformers"):
        return _predict_sentence_transformer(handle, body)
    elif arch_lower.startswith("yolov8"):
        return _predict_yolo(handle, body)
    else:
        raise HTTPException(status_code=400, detail=f"No predict handler for arch '{arch}'")


# ---------------------------------------------------------------------------
# APScheduler — 每 5 min refresh registry
# ---------------------------------------------------------------------------

_scheduler: Optional[BackgroundScheduler] = None


def _start_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        refresh_registry,
        trigger="interval",
        seconds=REGISTRY_REFRESH_SECONDS,
        id="registry_refresh",
        max_instances=1,
        coalesce=True,
    )
    sched.start()
    return sched


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    logger.info("Starting inference server (MAX_LOADED_MODELS=%d)", MAX_LOADED_MODELS)
    refresh_registry()
    _scheduler = _start_scheduler()
    _prom_set_loaded(0)
    try:
        yield
    finally:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)


app = FastAPI(
    title="ModelHub Inference Server",
    version="2.0.0",
    description="Multi-model lazy load + LRU + unified HTTP API (Phase 1: B+C)",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictBody(BaseModel):
    # xgboost-text / sentence-transformers
    texts: Optional[list[str]] = None
    # llamaguard-1b
    prompt: Optional[str] = None
    # yolov8
    image_url: Optional[str] = None
    image_b64: Optional[str] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "modelhub-inference",
        "version": "2.0.0",
        "loaded_models": list(_lru_cache.keys()),
        "registry_size": len(_registry),
        "max_loaded": MAX_LOADED_MODELS,
    }


@app.get("/v1/models")
def list_models():
    """列出已登記的所有 model（從 registry cache）。"""
    models = []
    for req_no, meta in _registry.items():
        models.append({
            "req_no": req_no,
            "model_name": meta.get("model_name", req_no),
            "arch": meta.get("arch", ""),
            "product": meta.get("product", ""),
            "status": meta.get("status", ""),
            "source": meta.get("source", ""),
            "in_ram": req_no in _lru_cache,
        })
    return {"models": models, "count": len(models)}


@app.get("/v1/models/{req_no}")
def get_model(req_no: str):
    """取得單一 model 詳情。"""
    key = req_no.upper()
    meta = _registry.get(key)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Model {req_no} not found in registry")
    return {
        **meta,
        "in_ram": key in _lru_cache,
    }


@app.get("/v1/models/{req_no}/loaded")
def is_loaded(req_no: str):
    """檢查 model 是否已在 RAM 中。"""
    key = req_no.upper()
    return {"req_no": key, "loaded": key in _lru_cache}


@app.post("/v1/models/{req_no}/unload")
def unload_model(req_no: str):
    """主動從 LRU cache 釋放 model。"""
    key = req_no.upper()
    if key not in _lru_cache:
        raise HTTPException(status_code=404, detail=f"Model {req_no} is not loaded")
    del _lru_cache[key]
    _prom_set_loaded(len(_lru_cache))
    logger.info("Manually unloaded %s, cache size=%d", key, len(_lru_cache))
    return {"req_no": key, "unloaded": True, "cache_size": len(_lru_cache)}


@app.post("/v1/predict")
async def predict(
    body: PredictBody,
    model: str = Query(..., description="req_no，如 MH-2026-022 或 EXT-2026-001"),
):
    """
    統一推論 endpoint。

    body 依 arch 而異：
      xgboost-text:          {"texts": [...]}
      llamaguard-1b:         {"prompt": "..."}
      sentence-transformers: {"texts": [...]}
      yolov8*:               {"image_url": "..."} 或 {"image_b64": "..."}

    response: {model, version, predictions, latency_ms}
    """
    req_no = model.upper()
    t0 = time.perf_counter()

    try:
        entry = _lazy_load(req_no)
    except HTTPException as e:
        _prom_count(req_no, "error")
        raise

    try:
        result = _run_predict(entry["arch"], entry["handle"], body.model_dump(exclude_unset=True))
    except HTTPException:
        _prom_count(req_no, "error")
        raise
    except Exception as e:
        _prom_count(req_no, "error")
        logger.error("Predict error for %s: %s", req_no, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    _prom_count(req_no, "ok")
    _prom_observe(req_no, latency_ms / 1000)

    return JSONResponse({
        "model": req_no,
        "version": entry["meta"].get("version", "latest"),
        "arch": entry["arch"],
        **result,
        "latency_ms": latency_ms,
    })


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint。"""
    if not _PROM_AVAILABLE:
        raise HTTPException(status_code=501, detail="prometheus_client not installed")
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
