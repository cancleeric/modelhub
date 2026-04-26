"""
backend/resources/kernel_registry.py — req_no 對應 Kaggle kernel 目錄

Sprint 15 P1-2: 靜態 KERNEL_MAP。
Sprint 17 P2-2: 改為動態掃描 kaggle-kernels/ 子目錄的 kernel-metadata.json，
               不再 hardcode dict。各 metadata 需有 "req_no" 欄位。
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("modelhub.resources.kernel_registry")

# Kaggle kernels 根目錄（相對於本檔往上兩層到 modelhub root）
_KAGGLE_KERNELS_ROOT = Path(__file__).parent.parent.parent / "kaggle-kernels"

# P2-18: module-level TTL cache，避免每次呼叫都掃磁碟
_cache: Optional[dict] = None
_cache_expires: float = 0.0


def _get_registry(ttl: int = 300) -> dict:
    """回傳 {req_no: kernel_dir_name}，TTL 300s（可透過參數覆寫）。"""
    global _cache, _cache_expires
    if _cache is None or time.monotonic() > _cache_expires:
        _cache = _scan_kernels()
        _cache_expires = time.monotonic() + ttl
    return _cache


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


def get_kernel_dir(req_no: str) -> Optional[Path]:
    """
    回傳 kaggle-kernels/{kernel_name}/ 的絕對路徑。
    若 req_no 不在掃描結果或目錄不存在，回傳 None。
    結果來自 TTL cache（預設 300s），不每次掃磁碟。
    """
    kernel_map = _get_registry()
    kernel_name = kernel_map.get(req_no)
    if not kernel_name:
        return None
    kernel_dir = _KAGGLE_KERNELS_ROOT / kernel_name
    if not kernel_dir.exists():
        return None
    return kernel_dir


def has_kaggle_kernel(req_no: str) -> bool:
    """該 req_no 是否有對應的 Kaggle kernel 目錄（結果來自 TTL cache）"""
    return get_kernel_dir(req_no) is not None


def scaffold_if_missing(submission) -> Optional[Path]:
    """
    M17-4 helper：若 req_no 對應的 kernel 目錄不存在，呼叫 KernelScaffolder 建立。
    建立後清除 TTL cache，讓下次 get_kernel_dir 能立即偵測到新目錄。
    成功回傳 kernel dir Path；失敗回傳 None（不拋例外）。
    """
    global _cache, _cache_expires

    req_no = getattr(submission, "req_no", None)
    if not req_no:
        logger.warning("scaffold_if_missing: submission 缺 req_no")
        return None

    # 先查 cache，若已存在直接回傳
    existing = get_kernel_dir(req_no)
    if existing:
        return existing

    try:
        from resources.kernel_scaffolder import KernelScaffolder
        scaffolder = KernelScaffolder()
        kernel_dir = scaffolder.scaffold(submission)
        # 清除 TTL cache，確保下次掃描能看到新目錄
        _cache = None
        _cache_expires = 0.0
        logger.info("scaffold_if_missing: scaffold OK req=%s dir=%s", req_no, kernel_dir)
        return kernel_dir
    except Exception as e:
        logger.warning("scaffold_if_missing: failed req=%s: %s", req_no, e)
        return None
