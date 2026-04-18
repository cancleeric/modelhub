"""
Lightning AI Job poller — Sprint 15 P2-1

每 120 秒 poll 所有 platform=lightning 且 status=training 的 submission：
- 狀態變化寫 submission_history
- complete 時下載 output → 解析 metrics → 建 ModelVersion
- error 時標記 training_failed
- >48 小時未完成 → 通知 CTO overtime

需要：LIGHTNING_API_KEY env var。
如果 env var 未設，poller 還是會啟動但會 log warning，並跳過 API 呼叫。

TODO: 所有標記 TODO 的區塊待取得 LIGHTNING_API_KEY 後補完。
      取得方式：https://lightning.ai → Settings → API Keys
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from models import SessionLocal, Submission, ModelVersion
from parsers import parse_training_log
from notifications import notify_event

logger = logging.getLogger("modelhub.poller.lightning")

POLL_INTERVAL_SECONDS = int(os.environ.get("MODELHUB_LIGHTNING_POLL_INTERVAL", "120"))
OVERTIME_HOURS = int(os.environ.get("MODELHUB_TRAINING_OVERTIME_HOURS", "48"))
LIGHTNING_DOWNLOAD_DIR = os.environ.get("MODELHUB_LIGHTNING_DL_DIR", "/tmp/modelhub-lightning")

# Lightning AI 計費（T4 免費 22hr/月，超量 $0.5/hr 估算）
GPU_USD_PER_HOUR = float(os.environ.get("MODELHUB_LIGHTNING_GPU_USD_PER_HOUR", "0.5"))
LIGHTNING_IS_FREE_TIER = os.environ.get("LIGHTNING_IS_FREE_TIER", "true").lower() == "true"


def _lightning_env_ready() -> bool:
    return bool(os.environ.get("LIGHTNING_API_KEY"))


def _get_lightning_studio(studio_name: str):
    """
    取得 Lightning Studio 實例（lazy import）。
    TODO: 待 API Key 後實作。
    """
    try:
        from lightning_sdk import Studio  # type: ignore
        username = os.environ.get("LIGHTNING_USERNAME", "boardgamegroup")
        teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "boardgamegroup")
        studio = Studio(name=studio_name, teamspace=teamspace)
        return studio
    except Exception as exc:
        logger.warning("Lightning SDK unavailable: %s", exc)
        return None


async def _fetch_job_status(studio_name: str) -> Optional[dict]:
    """
    查詢 Lightning Studio job 狀態。

    TODO: 待 API Key 後補完。
    預期回傳格式：
        {"status": "running"/"complete"/"error"/"queued", "raw": str}
    """
    if not _lightning_env_ready():
        logger.warning("LIGHTNING_API_KEY 未設，跳過 status fetch")
        return None

    # TODO: 實作
    # studio = _get_lightning_studio(studio_name)
    # if not studio:
    #     return None
    # status = studio.status  # "running" / "stopped" / "error" 等
    # return {"status": _normalize_status(status), "raw": str(status)}
    logger.debug("[TODO] _fetch_job_status studio=%s（待 API Key 實作）", studio_name)
    return None


async def _download_job_output(studio_name: str, dest_dir: str) -> Optional[str]:
    """
    下載 Lightning job output（result.json + best.pt 等）。

    TODO: 待 API Key 後補完。
    Lightning Studio 產出會在 /teamspace/studios/this_studio/kaggle/working/ 等路徑。
    """
    if not _lightning_env_ready():
        return None

    # TODO: 實作
    # studio = _get_lightning_studio(studio_name)
    # studio.download("result.json", dest_dir)
    logger.debug("[TODO] _download_job_output studio=%s（待 API Key 實作）", studio_name)
    return None


def _normalize_status(raw_status: str) -> str:
    """Lightning status 正規化為 modelhub 標準狀態"""
    mapping = {
        "running": "running",
        "stopped": "complete",
        "failed": "error",
        "error": "error",
        "queued": "queued",
        "pending": "queued",
        "starting": "queued",
    }
    return mapping.get(raw_status.lower(), raw_status.lower())


async def _process_submission(
    db: Session,
    submission: "Submission",
) -> None:
    """
    處理單一 Lightning submission 的 poll 週期。
    結構與 kaggle_poller._process_submission 一致。

    TODO: 完整邏輯待 API Key 後補完。
    """
    studio_name = getattr(submission, "kaggle_kernel_slug", None)
    if not studio_name:
        logger.warning("submission %s 無 studio_name，跳過", submission.id)
        return

    status_info = await _fetch_job_status(studio_name)
    if status_info is None:
        # API Key 未設或 TODO 尚未實作，靜默跳過
        return

    status = status_info.get("status", "unknown")
    logger.info("submission %s studio=%s status=%s", submission.id, studio_name, status)

    # 狀態變化寫 submission_history
    if submission.status != status:
        submission.status = status
        submission.updated_at = datetime.utcnow()
        db.commit()

    if status == "complete":
        # TODO: 下載 output → 解析 metrics → 建 ModelVersion
        dest_dir = str(Path(LIGHTNING_DOWNLOAD_DIR) / str(submission.id))
        output_dir = await _download_job_output(studio_name, dest_dir)
        if output_dir:
            # 與 kaggle_poller 相同的 parse + create ModelVersion 流程
            result_json_candidates = list(Path(output_dir).rglob("result.json"))
            if result_json_candidates:
                raw_log = result_json_candidates[0].read_text()
                metrics = parse_training_log(raw_log)
                logger.info("submission %s metrics=%s", submission.id, metrics)
                # TODO: create ModelVersion（參考 kaggle_poller 實作）
            else:
                logger.warning("submission %s 無 result.json，無法解析 metrics", submission.id)

    elif status == "error":
        submission.status = "training_failed"
        db.commit()
        await notify_event(
            event="training_failed",
            req_no=submission.req_no,
            message=f"Lightning job {studio_name} 失敗",
        )

    # Overtime 偵測
    created_at = getattr(submission, "created_at", None)
    if created_at and status in ("running", "queued"):
        elapsed = datetime.utcnow() - created_at
        if elapsed > timedelta(hours=OVERTIME_HOURS):
            logger.warning(
                "submission %s overtime! studio=%s elapsed=%.1fh",
                submission.id, studio_name, elapsed.total_seconds() / 3600,
            )
            await notify_event(
                event="training_overtime",
                req_no=submission.req_no,
                message=f"Lightning job {studio_name} 超過 {OVERTIME_HOURS}h 未完成",
            )


async def run_lightning_poller() -> None:
    """
    Lightning poller 主迴圈。結構與 kaggle_poller.run_poller 一致。

    TODO: 完整邏輯待 API Key 後補完。目前會啟動但跳過所有 API 呼叫。
    """
    logger.info("Lightning poller 啟動 (interval=%ds)", POLL_INTERVAL_SECONDS)
    if not _lightning_env_ready():
        logger.warning(
            "LIGHTNING_API_KEY 未設。poller 以 no-op 模式運行。\n"
            "取得方式：https://lightning.ai → Settings → API Keys"
        )

    while True:
        try:
            db: Session = SessionLocal()
            try:
                # TODO: 實際查詢 platform=lightning 的 submission
                # submissions = (
                #     db.query(Submission)
                #     .filter(
                #         Submission.platform == "lightning",
                #         Submission.status.in_(["training", "queued", "running"]),
                #     )
                #     .all()
                # )
                # for sub in submissions:
                #     await _process_submission(db, sub)
                logger.debug("[TODO] Lightning poller tick（待 API Key 後補 query）")
            finally:
                db.close()
        except Exception as exc:
            logger.exception("Lightning poller 迴圈例外: %s", exc)

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    asyncio.run(run_lightning_poller())
