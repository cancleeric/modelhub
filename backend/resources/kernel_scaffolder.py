"""
backend/resources/kernel_scaffolder.py — Kaggle Kernel 自動 Scaffold（M17-1/2）

輸入 Submission 物件，輸出 kaggle-kernels/<slug>/ 目錄：
  - kernel-metadata.json（slug/dataset_sources 由 submission 欄位自動生成）
  - train_kaggle.py（由 Jinja2 template 渲染）

slug 格式：boardgamegroup/mh-YYYY-NNN-<name>
  name 取自 req_name（小寫、空格換 -），最多 40 字元

防護：若目錄已存在，不覆寫（M17-4 風險規避）。
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("modelhub.resources.kernel_scaffolder")

# kaggle-kernels 根目錄（相對於本檔往上三層到 modelhub root）
_KAGGLE_KERNELS_ROOT = Path(__file__).parent.parent.parent / "kaggle-kernels"
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TEMPLATE_FILE = _TEMPLATES_DIR / "train_kaggle.py.j2"

# Kaggle 帳號（boardgamegroup）
KAGGLE_USERNAME = "boardgamegroup"

# YOLO arch 判斷（支援 yolo* 開頭）
_YOLO_ARCH_PATTERN = re.compile(r"^yolo", re.IGNORECASE)


def _slugify_name(name: str) -> str:
    """
    將 req_name 轉成 slug 片段：
    - 小寫
    - 非英數字元換成 -
    - 去掉連續 -
    - 最多 40 字元
    """
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name[:40]


def _make_slug(submission) -> str:
    """
    生成 Kaggle kernel slug：
    boardgamegroup/mh-YYYY-NNN-<name>

    req_no 格式：MH-2026-001 → 轉小寫 mh-2026-001
    req_name 有值時附加 -<name>，否則只用 req_no 部分。
    """
    req_no_lower = submission.req_no.lower()  # e.g. mh-2026-001
    if submission.req_name:
        name_part = _slugify_name(submission.req_name)
        kernel_name = f"{req_no_lower}-{name_part}"
    else:
        kernel_name = req_no_lower
    return f"{KAGGLE_USERNAME}/{kernel_name}"


def _extract_dataset_slug(dataset_path: Optional[str]) -> Optional[str]:
    """
    從 dataset_path 或 kaggle_dataset_url 提取 Kaggle dataset slug（最後一段）。
    e.g. "boardgamegroup/aicad-instrument-drawings" → "aicad-instrument-drawings"
         "/path/to/local/data" → None（本機路徑，不是 Kaggle dataset）
    """
    if not dataset_path:
        return None
    # 若包含 "kaggle.com" 或純 "user/dataset" 格式
    if "/" in dataset_path and not dataset_path.startswith("/"):
        return dataset_path.split("/")[-1]
    return None


def _get_dataset_sources(submission) -> list[str]:
    """
    從 submission 推導 dataset_sources（Kaggle dataset slug list）。
    優先從 kaggle_dataset_url 取，再從 dataset_path 取。
    無法推導時回傳空 list。
    """
    sources = []
    # 優先：kaggle_dataset_url（形如 "boardgamegroup/my-dataset"）
    kurl = getattr(submission, "kaggle_dataset_url", None)
    if kurl and "/" in kurl and not kurl.startswith("/"):
        sources.append(kurl.split("?")[0].strip())
        return sources
    # 次要：dataset_path 推導
    dp = getattr(submission, "dataset_path", None)
    slug = _extract_dataset_slug(dp)
    if slug:
        sources.append(f"{KAGGLE_USERNAME}/{slug}")
    return sources


def _get_epochs(submission) -> int:
    """從 submission 推導 epochs，無設定時預設 100。"""
    # submission 目前沒有 epochs 欄位，用 hyperparams 或環境預設值
    return 100


def _render_template(submission, dataset_slug: Optional[str]) -> str:
    """用 Jinja2 渲染 train_kaggle.py.j2。"""
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError:
        raise RuntimeError(
            "jinja2 not installed. Run: pip install jinja2"
        )

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("train_kaggle.py.j2")

    return tmpl.render(
        req_no=submission.req_no,
        req_name=getattr(submission, "req_name", None) or submission.req_no,
        arch=getattr(submission, "arch", None) or "yolov8m",
        epochs=_get_epochs(submission),
        class_count=getattr(submission, "class_count", None) or 1,
        dataset_slug=dataset_slug,
    )


class KernelScaffolder:
    """Kaggle Kernel 自動 Scaffold 工具"""

    def scaffold(self, submission) -> Path:
        """
        為給定 Submission 建立 kaggle-kernels/<dir>/ 目錄。

        目錄名稱取 kernel slug 的 kernel 部分（slash 後半段）。
        若目錄已存在，直接回傳既有路徑（不覆寫）。

        回傳建立（或既有）的 kernel 目錄 Path。
        raises: RuntimeError on template render failure
        """
        slug = _make_slug(submission)
        kernel_dir_name = slug.split("/")[-1]  # boardgamegroup/<this-part>
        kernel_dir = _KAGGLE_KERNELS_ROOT / kernel_dir_name

        # 防護：目錄已存在不覆寫
        if kernel_dir.exists():
            logger.info(
                "scaffold: kernel dir already exists, skip. req=%s dir=%s",
                submission.req_no, kernel_dir,
            )
            return kernel_dir

        kernel_dir.mkdir(parents=True, exist_ok=False)
        logger.info("scaffold: created dir %s for req=%s", kernel_dir, submission.req_no)

        # 生成 kernel-metadata.json
        dataset_sources = _get_dataset_sources(submission)
        dataset_slug = dataset_sources[0].split("/")[-1] if dataset_sources else None

        metadata = {
            "id": slug,
            "title": f"{submission.req_no} {getattr(submission, 'req_name', None) or ''} {getattr(submission, 'arch', 'yolov8m') or 'yolov8m'}".strip(),
            "code_file": "train_kaggle.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": dataset_sources,
            "competition_sources": [],
            "kernel_sources": [],
            "req_no": submission.req_no,
        }
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("scaffold: wrote kernel-metadata.json slug=%s", slug)

        # 渲染並寫入 train_kaggle.py
        try:
            script_content = _render_template(submission, dataset_slug)
        except Exception as e:
            # template 渲染失敗：移除已建目錄並重新 raise
            import shutil
            shutil.rmtree(kernel_dir, ignore_errors=True)
            raise RuntimeError(f"Template render failed for req={submission.req_no}: {e}") from e

        (kernel_dir / "train_kaggle.py").write_text(script_content, encoding="utf-8")
        logger.info("scaffold: wrote train_kaggle.py req=%s arch=%s",
                    submission.req_no, getattr(submission, "arch", "yolov8m"))

        return kernel_dir

    def scaffold_if_missing(self, submission) -> Optional[Path]:
        """
        若 kernel dir 不存在則 scaffold，否則回傳既有路徑。
        異常時記 warning 並回傳 None（不阻塞呼叫端）。
        """
        try:
            return self.scaffold(submission)
        except Exception as e:
            logger.warning(
                "scaffold_if_missing: failed for req=%s: %s",
                getattr(submission, "req_no", "?"), e,
            )
            return None
