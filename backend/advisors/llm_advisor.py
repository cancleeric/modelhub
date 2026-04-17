"""
Sprint 7.2 — LLM 輔助審核

走 Anemone LLM Gateway（集團規範 ADR-002）。
需要：
- ANEMONE_API_URL      例：http://anemone-api-dev:8920
- ANEMONE_API_TOKEN    modelhub 在 LIDS 註冊的 JWT（tenant_id=hurricanecore）

無法呼叫時（token 未設、Anemone 不可達、timeout、非 200）靜默回傳 []。
"""
import json
import logging
import os
from typing import List

import httpx

logger = logging.getLogger("modelhub.advisor")

ANEMONE_URL = os.getenv("ANEMONE_API_URL", "http://anemone-api-dev:8920").rstrip("/")
ANEMONE_TOKEN = os.getenv("ANEMONE_API_TOKEN", "")
ADVISOR_TIMEOUT = float(os.getenv("MODELHUB_ADVISOR_TIMEOUT", "10"))
ADVISOR_MODEL = os.getenv("MODELHUB_ADVISOR_MODEL", "gemini-2.5-pro")


def _build_prompt(payload) -> str:
    dataset_train_count = getattr(payload, "dataset_train_count", None)
    class_count = getattr(payload, "class_count", None)
    arch = getattr(payload, "arch", None)

    per_class_estimate: str
    if dataset_train_count and class_count and class_count > 0:
        avg = dataset_train_count / class_count
        per_class_estimate = f"{avg:.1f} images/class"
    else:
        per_class_estimate = "未知（dataset_train_count 或 class_count 未填）"

    fields = {
        "product": getattr(payload, "product", None),
        "company": getattr(payload, "company", None),
        "purpose": getattr(payload, "purpose", None),
        "model_type": getattr(payload, "model_type", None),
        "arch": arch,
        "class_list": getattr(payload, "class_list", None),
        "class_count": class_count,
        "dataset_train_count": dataset_train_count,
        "dataset_count": getattr(payload, "dataset_count", None),
        "dataset_val_count": getattr(payload, "dataset_val_count", None),
        "dataset_test_count": getattr(payload, "dataset_test_count", None),
        "map50_target": getattr(payload, "map50_target", None),
        "map50_95_target": getattr(payload, "map50_95_target", None),
        "inference_latency_ms": getattr(payload, "inference_latency_ms", None),
        "expected_delivery": getattr(payload, "expected_delivery", None),
        "priority": getattr(payload, "priority", None),
    }
    arch_eval_section = (
        f"\n\n[架構適配評估]\n"
        f"Per-class estimate: {per_class_estimate}\n"
        f"請評估此 arch ({arch}) 與 class_count ({class_count}) 及 per-class data 的匹配性，"
        f"若 per-class < 30，請明確建議補充資料或降低 class 數量。"
    )
    return (
        "以下是一張 ML 模型訓練需求單，請以資深 ML 工程師角度找出 3-5 個潛在風險或建議改善項。"
        "用繁體中文，每點一行，不要加序號或 bullet 符號，直接寫建議內容。"
        "重點關注：資料量與 mAP 目標是否合理、類別設計完整性、推論延遲與架構匹配、驗收標準明確性。\n\n"
        f"需求單內容:\n{json.dumps(fields, ensure_ascii=False, indent=2)}"
        f"{arch_eval_section}"
    )


async def review_submission(payload) -> List[str]:
    """回傳 suggestions list。Anemone 不可達 → []。"""
    if not ANEMONE_TOKEN:
        logger.debug("ANEMONE_API_TOKEN not set, skip LLM advisor")
        return []

    prompt = _build_prompt(payload)
    try:
        async with httpx.AsyncClient(timeout=ADVISOR_TIMEOUT) as client:
            resp = await client.post(
                f"{ANEMONE_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {ANEMONE_TOKEN}",
                    "X-Call-Type": "modelhub/submission-review",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ADVISOR_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "mode": "headless",
                },
            )
            if resp.status_code != 200:
                logger.info("LLM advisor non-200 (%d): %s",
                            resp.status_code, resp.text[:200])
                return []
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
            )
            if not content:
                return []
            # 一行一個建議
            lines = [ln.strip(" -•*\t") for ln in content.splitlines() if ln.strip()]
            return [ln for ln in lines if len(ln) > 5][:6]
    except Exception as e:
        logger.info("LLM advisor error: %s", e)
        return []
