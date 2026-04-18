"""YOLO log parser — 抓 mAP50 / mAP50-95 / epochs / best batch size / per-class AP50"""
import re


def parse_yolo_log(log_text: str) -> dict:
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
