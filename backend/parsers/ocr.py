"""OCR log parser — CER / WER / acc"""
import re


def parse_ocr_log(log_text: str) -> dict:
    metrics: dict = {}

    patterns = [
        ("cer", r"\bcer\s*[:=]\s*([\d.]+)"),
        ("wer", r"\bwer\s*[:=]\s*([\d.]+)"),
        ("val_accuracy", r"val[_\s]*acc(?:uracy)?\s*[:=]\s*([\d.]+)"),
    ]
    for key, pat in patterns:
        matches = re.findall(pat, log_text, flags=re.IGNORECASE)
        if matches:
            metrics[key] = float(matches[-1])

    epoch_m = re.findall(r"Epoch\s+(\d+)\s*/\s*(\d+)", log_text)
    if epoch_m:
        metrics["epochs"] = int(epoch_m[-1][1])

    return {"metrics": metrics, "raw_len": len(log_text)}
