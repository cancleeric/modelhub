"""
notifications.py — ModelHub 通知模組

commtool 是純 CLI 工具，無 HTTP API 模式。
使用 asyncio.create_subprocess_exec 呼叫，fire-and-forget。
失敗靜默處理，不影響主流程。
"""

import asyncio
import logging

logger = logging.getLogger("modelhub.notifications")


async def notify(to: str, message: str) -> None:
    """
    發送 CMC 私訊通知。

    Args:
        to: CMS username 或 email（commtool --to 支援兩者）
        message: 訊息內容
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "commtool", "msg", "send",
            "--to", to,
            "--msg", message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if proc.returncode != 0:
            logger.warning(
                "commtool notify failed (to=%s, rc=%d): %s",
                to, proc.returncode, stderr.decode(errors="replace"),
            )
        else:
            logger.info("commtool notify sent to %s", to)
    except asyncio.TimeoutError:
        logger.warning("commtool notify timeout (to=%s)", to)
    except FileNotFoundError:
        logger.warning("commtool not found in PATH, skipping notification (to=%s)", to)
    except Exception as e:
        logger.warning("commtool notify error (to=%s): %s", to, e)
