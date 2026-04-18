"""
backend/resources/kaggle_launcher.py — Kaggle kernel 推送與附掛（Sprint 15 P1-3）
"""

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from resources.kernel_registry import get_kernel_dir

logger = logging.getLogger("modelhub.resources.kaggle_launcher")


class KaggleLauncher:
    """負責把本地 kernel 目錄 push 到 Kaggle 並更新 submission 狀態"""

    def push_and_attach(self, req_no: str, db: Session, submission) -> bool:
        """
        1. 找 kernel 目錄（KernelRegistry.get_kernel_dir）
        2. kaggle kernels push -p {kernel_dir}
        3. 解析輸出取得 kernel_slug
        4. 更新 submission.kaggle_kernel_slug
        回傳成功/失敗
        """
        if not shutil.which("kaggle"):
            logger.error("kaggle CLI not found in PATH")
            return False

        kernel_dir = get_kernel_dir(req_no)
        if kernel_dir is None:
            logger.error("No kernel dir for req_no=%s", req_no)
            return False

        logger.info("Pushing kernel for req_no=%s from %s", req_no, kernel_dir)

        try:
            result = subprocess.run(
                ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            logger.error("kaggle kernels push timeout for req_no=%s", req_no)
            return False
        except Exception as e:
            logger.error("kaggle kernels push exception for req_no=%s: %s", req_no, e)
            return False

        if result.returncode != 0:
            logger.error(
                "kaggle kernels push failed for req_no=%s: %s",
                req_no,
                result.stderr.strip()[:500],
            )
            return False

        # 從輸出解析 kernel_slug（格式通常為 "Kernel version X successfully pushed."
        # 或含有 "username/kernel-name" 的行）
        kernel_slug = self._extract_slug(result.stdout, kernel_dir)

        if kernel_slug:
            submission.kaggle_kernel_slug = kernel_slug
            logger.info("Attached kernel_slug=%s to req_no=%s", kernel_slug, req_no)
        else:
            # 從 kernel-metadata.json 讀取 id 作為 fallback
            kernel_slug = self._read_slug_from_metadata(kernel_dir)
            if kernel_slug:
                submission.kaggle_kernel_slug = kernel_slug
                logger.info("Used metadata slug=%s for req_no=%s", kernel_slug, req_no)
            else:
                logger.warning("Could not determine kernel_slug for req_no=%s", req_no)
                # P3-3: slug 解析失敗寫 history
                try:
                    import json as _json
                    from models import SubmissionHistory
                    row = SubmissionHistory(
                        req_no=req_no,
                        action="slug_parse_failed",
                        actor="kaggle-launcher",
                        note=result.stdout[:500] if result.stdout else None,
                        meta=_json.dumps({"stderr": result.stderr[:200]}, ensure_ascii=False),
                    )
                    db.add(row)
                    db.commit()
                except Exception as hist_err:
                    logger.warning("Failed to write slug_parse_failed history: %s", hist_err)

        submission.kaggle_status = "queued"
        db.commit()
        return True

    def _extract_slug(self, output: str, kernel_dir: Path) -> Optional[str]:
        """從 kaggle push 輸出嘗試提取 slug，收窄到當前使用者名稱"""
        import os as _os
        username = _os.environ.get("KAGGLE_USERNAME", "")
        if username:
            pattern = rf'({re.escape(username)}/[a-z0-9_-]+)'
        else:
            pattern = r'([a-z0-9_-]+/[a-z0-9_-]+)'
        m = re.search(pattern, output)
        if m:
            return m.group(1)
        # fallback 讀 kernel-metadata.json
        return self._read_slug_from_metadata(kernel_dir)

    def _read_slug_from_metadata(self, kernel_dir: Path) -> Optional[str]:
        """從 kernel-metadata.json 讀 id 欄位作為 slug"""
        import json
        metadata_path = kernel_dir / "kernel-metadata.json"
        if not metadata_path.exists():
            return None
        try:
            data = json.loads(metadata_path.read_text())
            return data.get("id")
        except Exception:
            return None
