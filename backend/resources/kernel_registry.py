"""
backend/resources/kernel_registry.py — req_no 對應 Kaggle kernel 目錄

Sprint 15 P1-2: 靜態 KERNEL_MAP。
Sprint 17 P2-2: 改為動態掃描 kaggle-kernels/ 子目錄的 kernel-metadata.json，
               不再 hardcode dict。各 metadata 需有 "req_no" 欄位。
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("modelhub.resources.kernel_registry")

# Kaggle kernels 根目錄（相對於本檔往上兩層到 modelhub root）
_KAGGLE_KERNELS_ROOT = Path(__file__).parent.parent.parent / "kaggle-kernels"


def _scan_kernels() -> dict[str, str]:
    """
    掃描 kaggle-kernels/ 子目錄，讀取各目錄的 kernel-metadata.json。
    回傳 {req_no: kernel_dir_name} 的 dict。
    metadata 需有 "req_no" 欄位，否則該目錄略過。
    """
    result: dict[str, str] = {}
    if not _KAGGLE_KERNELS_ROOT.exists():
        logger.warning("kaggle-kernels 目錄不存在: %s", _KAGGLE_KERNELS_ROOT)
        return result

    for kernel_dir in sorted(_KAGGLE_KERNELS_ROOT.iterdir()):
        if not kernel_dir.is_dir():
            continue
        meta_path = kernel_dir / "kernel-metadata.json"
        if not meta_path.exists():
            continue
        try:
            data = json.loads(meta_path.read_text())
            req_no = data.get("req_no")
            if req_no:
                result[req_no] = kernel_dir.name
            else:
                logger.debug(
                    "kernel_dir=%s 的 kernel-metadata.json 缺 req_no 欄位，略過",
                    kernel_dir.name,
                )
        except Exception as e:
            logger.warning("讀取 %s 失敗: %s", meta_path, e)

    return result


# 動態建構（模組載入時掃描一次）；get_kernel_dir / has_kaggle_kernel 每次呼叫時重新掃描
# 若效能有疑慮，未來可加 TTL cache。


def get_kernel_dir(req_no: str) -> Optional[Path]:
    """
    回傳 kaggle-kernels/{kernel_name}/ 的絕對路徑。
    若 req_no 不在掃描結果或目錄不存在，回傳 None。
    """
    kernel_map = _scan_kernels()
    kernel_name = kernel_map.get(req_no)
    if not kernel_name:
        return None
    kernel_dir = _KAGGLE_KERNELS_ROOT / kernel_name
    if not kernel_dir.exists():
        return None
    return kernel_dir


def has_kaggle_kernel(req_no: str) -> bool:
    """該 req_no 是否有對應的 Kaggle kernel 目錄"""
    return get_kernel_dir(req_no) is not None
