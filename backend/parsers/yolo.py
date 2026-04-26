"""YOLO log parser — 抓 mAP50 / mAP50-95 / epochs / best batch size / per-class AP50

Kaggle kernel output 下載的 .log 是 NDJSON 格式（每行為
{"stream_name":"stdout","time":..,"data":"..."}），需先解碼才能餵給 regex。
parse_yolo_log 會自動偵測並解碼 NDJSON，再執行 regex 解析。
"""
import json
import re


def _decode_ndjson_log(text: str) -> str:
    """
    若 log 是 Kaggle NDJSON 格式，提取所有 stdout/stderr data 欄位，
    合成純文字 log 回傳。若不是 NDJSON，原樣回傳。

    Kaggle log 格式：每行是一個獨立 JSON object（第一行可能以 '[{' 開頭，
    其後以 ',{' 開頭），而不是標準 JSON array。例：
        [{"stream_name":"stdout","time":1.0,"data":"..."}
        ,{"stream_name":"stdout","time":2.0,"data":"..."}
        ]
    偵測方式：掃描前 20 行，若有任一行 JSON parse 成功且含 stream_name/data 欄位，
    則視為 NDJSON 並解碼所有行。
    """
    lines = text.splitlines()
    if not lines:
        return text

    # 偵測：掃前 20 行，找到一行含 stream_name+data 的 JSON 即確認為 NDJSON
    is_ndjson = False
    for line in lines[:20]:
        stripped = line.lstrip("[,]").strip()
        if not stripped.startswith("{"):
            continue
        try:
            obj = json.loads(stripped)
            if "stream_name" in obj and "data" in obj:
                is_ndjson = True
                break
        except (json.JSONDecodeError, TypeError):
            continue

    if not is_ndjson:
        return text  # 非 NDJSON，直接回傳

    plain_parts = []
    for line in lines:
        stripped = line.lstrip("[,]").strip()
        if not stripped.startswith("{"):
            continue
        try:
            obj = json.loads(stripped)
            if obj.get("stream_name") in ("stdout", "stderr"):
                plain_parts.append(obj.get("data", ""))
        except (json.JSONDecodeError, TypeError):
            continue
    return "".join(plain_parts)


def parse_yolo_log(log_text: str) -> dict:
    # 先嘗試從 result.json 直接取指標（kernel 標準輸出格式）
    # result.json 被 read_log_files 讀入後混在 log_text 中，掃描每行找 JSON
    _result_json_metrics: dict = {}
    _result_json_per_class: dict = {}
    for _line in log_text.splitlines():
        _stripped = _line.strip()
        if not _stripped.startswith("{"):
            continue
        try:
            _obj = json.loads(_stripped)
            if "map50" in _obj and "map50_95" in _obj:
                _result_json_metrics["map50"] = float(_obj["map50"])
                _result_json_metrics["map50_95"] = float(_obj["map50_95"])
                if "epochs" in _obj:
                    _result_json_metrics["epochs"] = int(_obj["epochs"])
                if "per_class_map50" in _obj and isinstance(_obj["per_class_map50"], dict):
                    _result_json_per_class = {
                        k: round(float(v), 6)
                        for k, v in _obj["per_class_map50"].items()
                    }
                break  # 找到後立即停止
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if _result_json_metrics:
        return {
            "metrics": _result_json_metrics,
            "raw_len": len(log_text),
            "per_class": _result_json_per_class or None,
            "_source": "result_json",
        }

    # result.json 不在此 log_text — 嘗試 NDJSON 解碼後用 regex 解析
    log_text = _decode_ndjson_log(log_text)

    metrics: dict = {}

    # `all       <imgs>     <inst>     P    R    mAP50  mAP50-95`
    # 最後一個 matching row 通常是最終 val metrics
    m = re.findall(
        r"^\s*all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s*$",
        log_text,
        flags=re.MULTILINE,
    )
    if m:
        last = m[-1]
        metrics["map50"] = float(last[0])
        metrics["map50_95"] = float(last[1])

    # epochs = "Epoch N/N"
    epoch_m = re.findall(r"Epoch\s+(\d+)\s*/\s*(\d+)", log_text)
    if epoch_m:
        metrics["epochs"] = int(epoch_m[-1][1])

    # batch_size
    bs_m = re.search(r"batch[\s_-]*size\s*[:=]?\s*(\d+)", log_text, flags=re.IGNORECASE)
    if bs_m:
        metrics["batch_size"] = int(bs_m.group(1))

    # Per-class AP50 解析
    # YOLO v8 val log 格式（非 `all` 行）：
    #   <ClassName>   <Images>  <Instances>   P       R    mAP50  mAP50-95
    #   component        450       1200    0.91    0.88    0.850    0.700
    # 同時支援帶前綴空格的格式（縮排不定）。
    per_class: dict = {}
    # 先找表頭位置，確認是 val metrics block（避免誤抓訓練中間的 epoch summary）
    # 抓所有非 `all` 的 class row（class 名不含數字開頭，至少 2 個字元）
    class_row_pattern = re.compile(
        r"^\s{2,}(?P<cls>[A-Za-z][A-Za-z0-9_\-]{1,})\s+"  # class name（至少 2 char，不以數字開頭）
        r"\d+\s+\d+\s+"                                     # Images  Instances
        r"[\d.]+\s+[\d.]+\s+"                               # P  R
        r"(?P<ap50>[\d.]+)\s+[\d.]+\s*$",                   # mAP50  mAP50-95
        re.MULTILINE,
    )
    # 取最後一個連續 class block（最後一輪 val 的結果）
    # 策略：找所有 match，取緊接在最後一個 `all` 行之後的那批
    all_row_positions = [m.start() for m in re.finditer(
        r"^\s*all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*$",
        log_text, re.MULTILINE,
    )]
    search_start = all_row_positions[-2] if len(all_row_positions) >= 2 else 0
    for cm in class_row_pattern.finditer(log_text, search_start):
        cls_name = cm.group("cls")
        ap50_val = float(cm.group("ap50"))
        per_class[cls_name] = round(ap50_val, 6)

    return {"metrics": metrics, "raw_len": len(log_text), "per_class": per_class or None}
