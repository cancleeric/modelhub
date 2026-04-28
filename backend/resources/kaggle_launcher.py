"""
backend/resources/kaggle_launcher.py — Kaggle kernel 推送與附掛（Sprint 15 P1-3）

mh-fix/dispatcher-force-rerun:
- push 前注入 MH_DISPATCH_TS comment，確保 Kaggle 建新 version（防 cache hit）
- 加 force_rerun: bool 參數，CTO 可手動 override 強制重跑
- 新增 get_current_kernel_version() 供外部比對版本
"""

import logging
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from resources.kernel_registry import get_kernel_dir

logger = logging.getLogger("modelhub.resources.kaggle_launcher")

# push 前注入的 comment marker（固定前綴，方便 grep 清舊行）
_TS_MARKER = "# MH_DISPATCH_TS:"


def _inject_dispatch_ts(script_path: Path, ts: str) -> None:
    """
    在 kernel script 第一行插入（或更新）dispatch timestamp comment。
    確保每次 push 的 code_file 都不同 → Kaggle 必然建新 version 並啟動新 run。
    若第一行已是舊 marker，則覆蓋；否則插到最頂端。
    """
    original = script_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    new_marker_line = f"{_TS_MARKER} {ts}\n"

    if lines and lines[0].startswith(_TS_MARKER):
        # 更新舊 marker
        lines[0] = new_marker_line
    else:
        lines.insert(0, new_marker_line)

    script_path.write_text("".join(lines), encoding="utf-8")
    logger.debug("_inject_dispatch_ts: updated %s → %s", script_path.name, ts)


def _remove_dispatch_ts(script_path: Path) -> None:
    """push 後移除 marker，恢復 clean working tree（不污染 Gitea repo）。"""
    try:
        original = script_path.read_text(encoding="utf-8")
        lines = original.splitlines(keepends=True)
        if lines and lines[0].startswith(_TS_MARKER):
            lines.pop(0)
            script_path.write_text("".join(lines), encoding="utf-8")
    except Exception as e:
        logger.warning("_remove_dispatch_ts: failed to clean %s: %s", script_path.name, e)


def get_current_kernel_version(kernel_slug: str) -> Optional[int]:
    """
    呼叫 `kaggle kernels status <slug>` 解析當前 version number。
    用於 push 前比對：若 new_version > old_version，確認 push 成功產生新 run。
    回傳 None 表示無法取得（CLI 不可用或解析失敗）。
    """
    if not shutil.which("kaggle"):
        return None
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        # 輸出樣例: "boardgamegroup/slug has status "complete" - Version 5"
        m = re.search(r'[Vv]ersion\s+(\d+)', result.stdout)
        if m:
            return int(m.group(1))
        return None
    except Exception as e:
        logger.debug("get_current_kernel_version(%s) failed: %s", kernel_slug, e)
        return None


class KaggleLauncher:
    """負責把本地 kernel 目錄 push 到 Kaggle 並更新 submission 狀態"""

    def push_and_attach(
        self,
        req_no: str,
        db: Session,
        submission,
        force_rerun: bool = False,
    ) -> bool:
        """
        1. 找 kernel 目錄（KernelRegistry.get_kernel_dir）
        2. 注入 MH_DISPATCH_TS comment → 保證 Kaggle 建新 version（防 cache hit）
           - force_rerun=True：強制注入（CTO 手動重跑 stuck submission 用）
           - force_rerun=False（預設）：同樣注入，因為每次 push 都應觸發新 run
        3. kaggle kernels push -p {kernel_dir}
        4. 清除注入的 marker（還原 working tree）
        5. 解析輸出取得 kernel_slug
        6. 更新 submission.kaggle_kernel_slug / kaggle_status
        回傳成功/失敗
        """
        if not shutil.which("kaggle"):
            logger.error("kaggle CLI not found in PATH")
            return False

        kernel_dir = get_kernel_dir(req_no)
        if kernel_dir is None:
            logger.error("No kernel dir for req_no=%s", req_no)
            return False

        # 找 code_file（從 kernel-metadata.json 讀，fallback train_kaggle.py）
        import json as _json
        meta_path = kernel_dir / "kernel-metadata.json"
        code_file_name = "train_kaggle.py"
        if meta_path.exists():
            try:
                meta = _json.loads(meta_path.read_text())
                code_file_name = meta.get("code_file", code_file_name)
            except Exception:
                pass
        script_path = kernel_dir / code_file_name

        # 注入 dispatch timestamp → 確保 Kaggle 建新 version，不 cache
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dispatched_version_before: Optional[int] = None
        if script_path.exists():
            # 記錄 push 前版本（供 log 比對）
            slug_for_version = self._read_slug_from_metadata(kernel_dir)
            if slug_for_version:
                dispatched_version_before = get_current_kernel_version(slug_for_version)
            _inject_dispatch_ts(script_path, ts)
            logger.info(
                "push_and_attach: req=%s injected dispatch_ts=%s (force_rerun=%s, version_before=%s)",
                req_no, ts, force_rerun, dispatched_version_before,
            )
        else:
            logger.warning(
                "push_and_attach: script_path=%s not found, proceeding without ts injection",
                script_path,
            )

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
            _remove_dispatch_ts(script_path)
            return False
        except Exception as e:
            logger.error("kaggle kernels push exception for req_no=%s: %s", req_no, e)
            _remove_dispatch_ts(script_path)
            return False
        finally:
            # 無論 push 結果如何，恢復 working tree（cleanup）
            if script_path.exists():
                _remove_dispatch_ts(script_path)

        if result.returncode != 0:
            logger.error(
                "kaggle kernels push failed for req_no=%s: %s",
                req_no,
                result.stderr.strip()[:500],
            )
            return False

        # 從輸出解析 kernel_slug
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
                    import json as _json2
                    from models import SubmissionHistory
                    row = SubmissionHistory(
                        req_no=req_no,
                        action="slug_parse_failed",
                        actor="kaggle-launcher",
                        note=result.stdout[:500] if result.stdout else None,
                        meta=_json2.dumps({"stderr": result.stderr[:200]}, ensure_ascii=False),
                    )
                    db.add(row)
                    db.commit()
                except Exception as hist_err:
                    logger.warning("Failed to write slug_parse_failed history: %s", hist_err)

        submission.kaggle_status = "queued"
        db.commit()

        # 記錄 push 完成 log（含 ts marker，供事後 audit）
        logger.info(
            "push_and_attach done: req=%s slug=%s dispatch_ts=%s version_before=%s",
            req_no, submission.kaggle_kernel_slug, ts, dispatched_version_before,
        )
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
