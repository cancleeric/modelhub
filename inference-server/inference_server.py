"""
ModelHub 推論 Server — 跑在 host Python（port 8951）

支援模型：
  MH-2026-010  MobileNetV2  96×96   11-class PID 符號分類
  MH-2026-011  EfficientNet-B0  224×224  4-class 品質路由

API:
  GET  /health
  GET  /models          列出已載入的模型清單
  POST /predict/{req_no}  上傳圖片 → prediction + confidence + all_scores
"""

import io
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
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

TRAINING_BASE = Path("/Users/yinghaowang/HurricaneCore/modelhub/training")

# --- 模型設定表 ---
# key = req_no（小寫）
MODEL_CONFIGS = {
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


def _load_model_cached(req_no_lower: str) -> dict:
    """Lazy load：第一次呼叫時載入，後續從快取取。"""
    if req_no_lower in _loaded_models:
        return _loaded_models[req_no_lower]

    cfg = MODEL_CONFIGS.get(req_no_lower)
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
    """列出所有 verdict=pass 的可用模型。"""
    models = []
    for key, cfg in MODEL_CONFIGS.items():
        result_path = cfg["result_path"]
        if not result_path.exists():
            continue
        try:
            result = _load_result(result_path)
        except Exception:
            continue
        if result.get("verdict") != "pass":
            continue
        models.append({
            "req_no": cfg["req_no"],
            "name": cfg["name"],
            "classes": result["classes"],
            "accuracy": result.get("best_val_acc"),
        })
    return {"models": models}


@app.post("/predict/{req_no}")
async def predict(req_no: str, file: UploadFile = File(...)):
    """上傳圖片，回傳 prediction + confidence + all_scores。"""
    key = req_no.lower()
    entry = _load_model_cached(key)  # 404 if not available

    # 讀圖
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    # 前處理 + 推論
    tensor = entry["tf"](img).unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        logits = entry["model"](tensor)
        probs = F.softmax(logits, dim=1)[0]  # (num_classes,)

    classes = entry["classes"]
    pred_idx = probs.argmax().item()
    confidence = round(probs[pred_idx].item(), 6)
    prediction = classes[pred_idx]
    all_scores = {cls: round(probs[i].item(), 6) for i, cls in enumerate(classes)}

    cfg = entry["cfg"]
    result = entry["result"]
    return JSONResponse({
        "req_no": cfg["req_no"],
        "model": cfg["arch"],
        "prediction": prediction,
        "confidence": confidence,
        "all_scores": all_scores,
        # 額外資訊
        "accuracy": result.get("best_val_acc"),
    })


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("MODELHUB_INFERENCE_PORT", "8951"))
    logger.info("Starting inference server on port %d", port)
    logger.info("Available models: %s", list(MODEL_CONFIGS.keys()))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
