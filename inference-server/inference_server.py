"""
ModelHub 推論 Server — 跑在 host Python（port 8951）

支援模型：
  MH-2026-010  MobileNetV2  96×96   11-class PID 符號分類
  MH-2026-011  EfficientNet-B0  224×224  4-class 品質路由
  YOLO 偵測模型（ultralytics）

API:
  GET  /health
  GET  /models          列出已載入的模型清單
  POST /predict/{req_no}  上傳圖片 → prediction + confidence + all_scores（分類）
                                    或 detections JSON（偵測）
  query param: model_type=classification|detection（預設由 MODEL_CONFIGS 決定）
"""

import io
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    efficientnet_b0,
    mobilenet_v2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("inference_server")

import os
import httpx

TRAINING_BASE = Path(os.environ.get("MODELHUB_TRAINING_BASE", "/Users/yinghaowang/HurricaneCore/modelhub/training"))

# Registry API endpoint（同 network 內走容器名，本機開發走 localhost）
REGISTRY_API_URL = os.environ.get("MODELHUB_REGISTRY_URL", "http://modelhub-api-dev:8000/api/registry")
REGISTRY_API_FALLBACK = "http://localhost:8950/api/registry"
REGISTRY_API_KEY = os.environ.get("MODELHUB_API_KEY", "")

# --- 靜態模型設定表（fallback，當 registry API 不可達時使用）---
# key = req_no（小寫）
MODEL_CONFIGS: dict[str, dict] = {
    "mh-2026-010": {
        "req_no": "MH-2026-010",
        "name": "PID 符號分類",
        "arch": "mobilenetv2",
        "weight_path": TRAINING_BASE / "mh-2026-010" / "pid_symbols_best.pth",
        "result_path": TRAINING_BASE / "mh-2026-010" / "result.json",
        "img_size": 96,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
    },
    "mh-2026-011": {
        "req_no": "MH-2026-011",
        "name": "品質路由",
        "arch": "efficientnet_b0",
        "weight_path": TRAINING_BASE / "mh-2026-011" / "quality_router_best.pth",
        "result_path": TRAINING_BASE / "mh-2026-011" / "result.json",
        "img_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
}

# --- 執行期快取 ---
_loaded_models: dict[str, dict] = {}  # req_no_lower → {model, tf, classes, result}
# 動態 registry 快取（從 API 拉取，支援熱更新）
_dynamic_registry: dict[str, dict] = {}  # req_no_lower → registry entry


def _fetch_registry_from_api() -> list[dict]:
    """從 modelhub registry API 拉取所有 is_current=true 的模型清單。失敗回空 list。"""
    headers = {}
    if REGISTRY_API_KEY:
        headers["X-Api-Key"] = REGISTRY_API_KEY
    for url in [REGISTRY_API_URL, REGISTRY_API_FALLBACK]:
        try:
            resp = httpx.get(url, params={"is_current": "true"}, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("Registry API unreachable at %s: %s", url, e)
    logger.warning("Cannot reach registry API, using static MODEL_CONFIGS only")
    return []


def refresh_dynamic_registry() -> None:
    """從 API 拉取最新 current 模型清單，更新 _dynamic_registry。"""
    global _dynamic_registry
    entries = _fetch_registry_from_api()
    new_registry: dict[str, dict] = {}
    for entry in entries:
        req_no_lower = (entry.get("req_no") or "").lower()
        if not req_no_lower:
            continue
        file_path = entry.get("file_path") or entry.get("model_output_path")
        arch = (entry.get("arch") or "").lower()
        new_registry[req_no_lower] = {
            "req_no": entry.get("req_no", req_no_lower.upper()),
            "name": entry.get("model_name", req_no_lower),
            "arch": arch,
            "weight_path": Path(file_path) if file_path else None,
            "result_path": None,  # registry 不回傳 result.json，classification 需手動管理
            "img_size": 224,      # 預設值，可從 entry 擴充
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "_from_registry": True,
        }
    if new_registry:
        _dynamic_registry = new_registry
        logger.info(
            "Dynamic registry refreshed: %d models (%s)",
            len(new_registry), list(new_registry.keys()),
        )


def _get_model_config(req_no_lower: str) -> dict | None:
    """
    查詢模型設定：優先 dynamic registry（熱更新），fallback 靜態 MODEL_CONFIGS。
    每次呼叫 predict 時順帶嘗試更新 dynamic registry（非阻塞：失敗不影響推論）。
    """
    # 非同步 refresh：每次推論都嘗試刷新（低成本：5s timeout，失敗 silent）
    try:
        refresh_dynamic_registry()
    except Exception:
        pass
    # Dynamic registry 優先
    if req_no_lower in _dynamic_registry:
        return _dynamic_registry[req_no_lower]
    # Fallback 靜態設定
    return MODEL_CONFIGS.get(req_no_lower)


def _load_result(result_path: Path) -> dict:
    with open(result_path) as f:
        return json.load(f)


def _build_model(arch: str, num_classes: int) -> torch.nn.Module:
    if arch == "mobilenetv2":
        import torch.nn as nn
        m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif arch == "efficientnet_b0":
        import torch.nn as nn
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown arch: {arch}")


def _predict_yolo(model_path: Path, img_bytes: bytes) -> dict:
    """YOLO 推論（ultralytics）。回傳 detections JSON。"""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="ultralytics 套件未安裝，無法執行 YOLO 推論",
        )
    model = YOLO(str(model_path))
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = model(img, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
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
    return {"detections": detections, "count": len(detections)}


def _load_model_cached(req_no_lower: str) -> dict:
    """Lazy load：第一次呼叫時載入，後續從快取取。"""
    if req_no_lower in _loaded_models:
        return _loaded_models[req_no_lower]

    cfg = _get_model_config(req_no_lower)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Model not available for prediction")

    result = _load_result(cfg["result_path"])
    if result.get("verdict") != "pass":
        raise HTTPException(status_code=404, detail="Model not available for prediction")

    classes = result["classes"]
    num_classes = len(classes)

    logger.info("Loading model %s (%s, %d classes) from %s",
                cfg["req_no"], cfg["arch"], num_classes, cfg["weight_path"])

    model = _build_model(cfg["arch"], num_classes)
    state = torch.load(cfg["weight_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["normalize_mean"], cfg["normalize_std"]),
    ])

    _loaded_models[req_no_lower] = {
        "model": model,
        "tf": tf,
        "classes": classes,
        "result": result,
        "cfg": cfg,
    }
    logger.info("Model %s loaded OK", cfg["req_no"])
    return _loaded_models[req_no_lower]


# --- FastAPI app ---
app = FastAPI(title="ModelHub Inference Server", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "modelhub-inference", "loaded": list(_loaded_models.keys())}


@app.get("/models")
def list_models():
    """列出所有可用模型（動態 registry + verdict=pass 的靜態設定）。"""
    # 先嘗試刷新動態 registry
    try:
        refresh_dynamic_registry()
    except Exception:
        pass

    seen: set[str] = set()
    models = []

    # 動態 registry（來自 API）
    for key, cfg in _dynamic_registry.items():
        models.append({
            "req_no": cfg["req_no"],
            "name": cfg["name"],
            "arch": cfg.get("arch"),
            "source": "registry",
        })
        seen.add(key)

    # 靜態設定（fallback，排除已被 dynamic registry 覆蓋的）
    for key, cfg in MODEL_CONFIGS.items():
        if key in seen:
            continue
        result_path = cfg.get("result_path")
        if result_path is None or not Path(result_path).exists():
            continue
        try:
            result = _load_result(Path(result_path))
        except Exception:
            continue
        if result.get("verdict") != "pass":
            continue
        models.append({
            "req_no": cfg["req_no"],
            "name": cfg["name"],
            "arch": cfg.get("arch"),
            "classes": result["classes"],
            "accuracy": result.get("best_val_acc"),
            "source": "static",
        })

    return {"models": models}


@app.post("/predict/{req_no}")
async def predict(
    req_no: str,
    file: UploadFile = File(...),
    model_type: str = Query(default="", description="classification | detection（留空由設定決定）"),
):
    """
    上傳圖片，根據 model_type 走不同推論分支：
    - classification：回傳 prediction + confidence + all_scores
    - detection：回傳 detections JSON（boxes, scores, labels）
    """
    key = req_no.lower()
    contents = await file.read()

    # 決定 model_type：query param > MODEL_CONFIGS/dynamic registry 設定 > 預設 classification
    cfg = _get_model_config(key)
    resolved_type = model_type.strip().lower()
    if not resolved_type and cfg:
        cfg_arch = (cfg.get("arch") or "").lower()
        if cfg_arch.startswith("yolo"):
            resolved_type = "detection"
        else:
            resolved_type = "classification"
    if not resolved_type:
        resolved_type = "classification"

    # --- YOLO 偵測分支 ---
    if resolved_type == "detection":
        if cfg is None:
            raise HTTPException(status_code=404, detail="Model not available for prediction")
        weight_path = cfg.get("weight_path")
        if not weight_path or not Path(weight_path).exists():
            raise HTTPException(status_code=404, detail="Model weight file not found")
        result_data = _predict_yolo(Path(weight_path), contents)
        return JSONResponse({
            "req_no": cfg["req_no"],
            "model": cfg.get("arch", "yolo"),
            "model_type": "detection",
            **result_data,
        })

    # --- 分類分支（MobileNetV2 / EfficientNet）---
    entry = _load_model_cached(key)  # 404 if not available

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    tensor = entry["tf"](img).unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        logits = entry["model"](tensor)
        probs = F.softmax(logits, dim=1)[0]  # (num_classes,)

    classes = entry["classes"]
    pred_idx = probs.argmax().item()
    confidence = round(probs[pred_idx].item(), 6)
    prediction = classes[pred_idx]
    all_scores = {cls: round(probs[i].item(), 6) for i, cls in enumerate(classes)}

    entry_cfg = entry["cfg"]
    result = entry["result"]
    return JSONResponse({
        "req_no": entry_cfg["req_no"],
        "model": entry_cfg["arch"],
        "model_type": "classification",
        "prediction": prediction,
        "confidence": confidence,
        "all_scores": all_scores,
        "accuracy": result.get("best_val_acc"),
    })


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("MODELHUB_INFERENCE_PORT", "8951"))
    logger.info("Starting inference server on port %d", port)
    logger.info("Available models: %s", list(MODEL_CONFIGS.keys()))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
