"""XGBoost / text classifier result.json parser.

result.json 格式（MH-2026-022 範例）：
{
    "cv_accuracy_mean": 0.9662,
    "cv_f1_weighted_mean": 0.9673,
    "pass_fail": "pass",
    "accuracy": 0.9662,
    "map50": null,   ← text classifier 無此值
    ...
}

映射規則：
  map50 ← accuracy（cv_accuracy_mean 或 accuracy，取最高）
  map50_95 ← f1_weighted（cv_f1_weighted_mean）
  pass_fail ← pass_fail
"""
import json
import re
from typing import Optional


def _extract_result_json(log_text: str) -> Optional[dict]:
    """從 log 中提取 ##RESULT_JSON## 標記或直接解析 JSON。"""
    marker = "##RESULT_JSON##"
    idx = log_text.find(marker)
    if idx != -1:
        rest = log_text[idx + len(marker):]
        end_idx = rest.find(marker)
        if end_idx != -1:
            json_str = rest[:end_idx].strip()
        else:
            json_str = rest.strip()
        try:
            return json.loads(json_str)
        except Exception:
            pass

    # fallback: 嘗試從 stdout 行找 JSON
    for line in log_text.splitlines():
        line = line.strip()
        if line.startswith("{") and "cv_accuracy_mean" in line:
            try:
                return json.loads(line)
            except Exception:
                pass
    return None


def parse_xgboost_text_log(log_text: str) -> dict:
    """解析 XGBoost/text classifier 的 log，映射 accuracy → map50。"""
    result_obj = _extract_result_json(log_text)
    metrics: dict = {}

    if result_obj:
        # accuracy → map50（通用精度指標代理）
        acc = result_obj.get("cv_accuracy_mean") or result_obj.get("accuracy")
        if acc is not None:
            metrics["map50"] = float(acc)

        # f1_weighted → map50_95（相近語義）
        f1 = result_obj.get("cv_f1_weighted_mean")
        if f1 is not None:
            metrics["map50_95"] = float(f1)

        # pass_fail
        pf = result_obj.get("pass_fail")
        if pf:
            metrics["pass_fail"] = pf

        # 保留 accuracy/f1 原始值供參考
        if acc is not None:
            metrics["accuracy"] = float(acc)
        if f1 is not None:
            metrics["f1_weighted"] = float(f1)

    return {"metrics": metrics, "raw_len": len(log_text)}
