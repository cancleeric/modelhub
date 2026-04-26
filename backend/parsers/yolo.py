"""YOLO log parser — 抓 mAP50 / mAP50-95 / epochs / best batch size / per-class AP50
以及逐 epoch 訓練曲線（epoch_data）

Kaggle kernel output 下載的 .log 是 NDJSON 格式（每行為
{"stream_name":"stdout","time":..,"data":"..."}），需先解碼才能餵給 regex。
parse_yolo_log 會自動偵測並解碼 NDJSON，再執行 regex 解析。

epoch_data 解析策略（優先序）：
1. results.csv（YOLOv8 標準輸出，每 epoch 一行，含 train/val loss 和 map 指標）
2. YOLOv8 stdout 逐 epoch print（格式：Epoch N/N ... mAP50 ...）
"""
import csv
import io
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


def parse_results_csv(csv_text: str) -> list[dict]:
    """
    解析 YOLOv8 訓練輸出的 results.csv，回傳逐 epoch dict list。

    YOLOv8 results.csv 欄位（含前綴空格）：
        epoch, train/box_loss, train/cls_loss, train/dfl_loss,
        metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
        val/box_loss, val/cls_loss, val/dfl_loss, lr/pg0, lr/pg1, lr/pg2

    回傳格式：
        [{"epoch": 1, "train_loss": 0.8, "val_loss": 0.5, "map50": 0.3, "map50_95": 0.15}, ...]

    train_loss = train/box_loss + train/cls_loss（兩項合計，代表訓練總 loss）
    val_loss   = val/box_loss + val/cls_loss
    """
    if not csv_text or not csv_text.strip():
        return []

    # YOLOv8 results.csv 的欄位名含前綴空格，用 csv.DictReader strip key
    reader = csv.DictReader(io.StringIO(csv_text))
    # normalize key：去前後空白
    epoch_data: list[dict] = []
    try:
        for row in reader:
            norm = {k.strip(): v.strip() for k, v in row.items() if k is not None}
            epoch_val = norm.get("epoch")
            if epoch_val is None:
                continue
            try:
                epoch_int = int(float(epoch_val))
            except (ValueError, TypeError):
                continue

            def _f(key: str) -> float | None:
                v = norm.get(key)
                if v is None:
                    return None
                try:
                    return round(float(v), 6)
                except (ValueError, TypeError):
                    return None

            # box_loss + cls_loss 作為簡化 train_loss / val_loss
            train_box = _f("train/box_loss") or 0.0
            train_cls = _f("train/cls_loss") or 0.0
            val_box = _f("val/box_loss") or 0.0
            val_cls = _f("val/cls_loss") or 0.0

            entry: dict = {
                "epoch": epoch_int,
                "train_loss": round(train_box + train_cls, 6),
                "val_loss": round(val_box + val_cls, 6),
            }
            map50 = _f("metrics/mAP50(B)")
            if map50 is not None:
                entry["map50"] = map50
            map50_95 = _f("metrics/mAP50-95(B)")
            if map50_95 is not None:
                entry["map50_95"] = map50_95
            precision = _f("metrics/precision(B)")
            if precision is not None:
                entry["precision"] = precision
            recall = _f("metrics/recall(B)")
            if recall is not None:
                entry["recall"] = recall

            epoch_data.append(entry)
    except Exception:
        return []

    return epoch_data


def _parse_epoch_data_from_stdout(log_text: str) -> list[dict]:
    """
    從 YOLOv8 stdout 解析逐 epoch 資料（fallback，results.csv 不存在時使用）。

    YOLOv8 stdout 格式（每 epoch 一行）：
        Epoch 1/100  box_loss  cls_loss  dfl_loss  Instances  Size
                       0.800    1.200    1.000        50     640
        後接一行 val summary：
        all  N  N  P  R  mAP50  mAP50-95

    此函式只取 Epoch 行 + 對應 val `all` 行，組合成 epoch_data。
    """
    epoch_data: list[dict] = []

    # 找所有 "Epoch N/M" 行
    epoch_pat = re.compile(r"Epoch\s+(\d+)/(\d+)", re.MULTILINE)
    # 找所有 val `all` 行（final val metrics per epoch）
    val_pat = re.compile(
        r"^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
        re.MULTILINE,
    )

    epoch_matches = list(epoch_pat.finditer(log_text))
    val_matches = list(val_pat.finditer(log_text))

    if not epoch_matches or not val_matches:
        return []

    # 為每個 epoch，找其後第一個 val `all` 行
    val_idx = 0
    for em in epoch_matches:
        epoch_num = int(em.group(1))
        # 跳過已用的 val rows
        while val_idx < len(val_matches) and val_matches[val_idx].start() < em.start():
            val_idx += 1
        if val_idx >= len(val_matches):
            break
        vm = val_matches[val_idx]
        val_idx += 1

        try:
            entry: dict = {
                "epoch": epoch_num,
                "map50": round(float(vm.group(3)), 6),
                "map50_95": round(float(vm.group(4)), 6),
                "precision": round(float(vm.group(1)), 6),
                "recall": round(float(vm.group(2)), 6),
            }
            epoch_data.append(entry)
        except (ValueError, IndexError):
            continue

    return epoch_data


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
            "epoch_data": [],  # result.json 路徑沒有逐 epoch 資料，由 poller 從 results.csv 補充
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

    # 逐 epoch 資料：優先從 stdout 解析（results.csv 由 poller 直接呼叫 parse_results_csv）
    epoch_data = _parse_epoch_data_from_stdout(log_text)

    return {
        "metrics": metrics,
        "raw_len": len(log_text),
        "per_class": per_class or None,
        "epoch_data": epoch_data,
    }
