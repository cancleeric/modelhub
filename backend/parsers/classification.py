"""Classification log parser — val_accuracy / top-k"""
import re


def parse_classification_log(log_text: str) -> dict:
    metrics: dict = {}

    patterns = [
        ("val_accuracy", r"val[_\s]*acc(?:uracy)?\s*[:=]\s*([\d.]+)"),
        ("val_loss",     r"val[_\s]*loss\s*[:=]\s*([\d.]+)"),
        ("top1",         r"top[-_]?1\s*[:=]\s*([\d.]+)"),
        ("top5",         r"top[-_]?5\s*[:=]\s*([\d.]+)"),
    ]
    for key, pat in patterns:
        matches = re.findall(pat, log_text, flags=re.IGNORECASE)
        if matches:
            metrics[key] = float(matches[-1])

    epoch_m = re.findall(r"Epoch\s+(\d+)\s*/\s*(\d+)", log_text)
    if epoch_m:
        metrics["epochs"] = int(epoch_m[-1][1])

    return {"metrics": metrics, "raw_len": len(log_text)}
