"""
backend/resources/kernel_registry.py — req_no 對應 Kaggle kernel 目錄（Sprint 15 P1-2）
"""

from pathlib import Path
from typing import Optional

# Kaggle kernels 根目錄（相對於本檔往上兩層到 modelhub root）
_KAGGLE_KERNELS_ROOT = Path(__file__).parent.parent.parent / "kaggle-kernels"

# req_no → kaggle-kernels 子目錄名稱
KERNEL_MAP: dict[str, str] = {
    "MH-2026-005": "instrument_drawings_kernel",
    "MH-2026-006": "text_detection_kernel",       # Sprint 15: 移至 Kaggle GPU
    "MH-2026-008": "multiview_boundary_kernel",    # Sprint 15: 移至 Kaggle GPU
    "MH-2026-009": "ocr_kernel",
    "MH-2026-010": "pid_symbols_kernel",
    "MH-2026-011": "quality_router_kernel",
}


def get_kernel_dir(req_no: str) -> Optional[Path]:
    """
    回傳 kaggle-kernels/{kernel_name}/ 的絕對路徑。
    若 req_no 不在 KERNEL_MAP 或目錄不存在，回傳 None。
    """
    kernel_name = KERNEL_MAP.get(req_no)
    if not kernel_name:
        return None
    kernel_dir = _KAGGLE_KERNELS_ROOT / kernel_name
    if not kernel_dir.exists():
        return None
    return kernel_dir


def has_kaggle_kernel(req_no: str) -> bool:
    """該 req_no 是否有對應的 Kaggle kernel 目錄"""
    return get_kernel_dir(req_no) is not None
