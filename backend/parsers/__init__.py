"""
parsers registry — 由 arch 字首 match，回傳 dict 指標
"""
from .yolo import parse_yolo_log
from .classification import parse_classification_log
from .ocr import parse_ocr_log


def parse_training_log(arch: str | None, log_text: str) -> dict:
    """依 arch 決定解析器；回傳 dict，至少包含 metrics / raw 欄位。"""
    arch = (arch or "").lower()
    if arch.startswith("yolo"):
        return parse_yolo_log(log_text)
    if (
        arch.startswith("resnet")
        or arch.startswith("efficientnet")
        or arch.startswith("mobilenet")
        or arch.startswith("vit")
        or arch.startswith("cls")
        or arch.startswith("classification")
    ):
        return parse_classification_log(log_text)
    if "ocr" in arch or "crnn" in arch or "trocr" in arch:
        return parse_ocr_log(log_text)
    return parse_yolo_log(log_text)
