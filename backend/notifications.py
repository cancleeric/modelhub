"""
notifications.py — ModelHub 通知模組

- notify(to, message)：低階 CMC 私訊（commtool CLI，fire-and-forget）
- notify_event(event, submission, ...)：高階事件封裝（新建、核准、退件、完成、失敗、超時）
- send_email(to_list, subject, body)：透過 mailtool 寄 email（週報用）
- _dispatch_brain_webhook(event_type, submission, **kwargs)：M12 N4 跨服通報
"""

import asyncio
import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger("modelhub.notifications")


# ---------------------------------------------------------------------------
# 低階呼叫
# ---------------------------------------------------------------------------

async def notify(to: str, message: str) -> bool:
    """發送 CMC 私訊。回傳 True=成功；失敗不 raise。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "commtool", "msg", "send",
            "--to", to,
            "--msg", message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if proc.returncode != 0:
            logger.warning("commtool notify fail (to=%s rc=%d): %s",
                           to, proc.returncode, stderr.decode(errors="replace"))
            return False
        logger.info("commtool notify sent to %s", to)
        return True
    except asyncio.TimeoutError:
        logger.warning("commtool notify timeout (to=%s)", to)
    except FileNotFoundError:
        logger.warning("commtool not found in PATH, skip (to=%s)", to)
    except Exception as e:
        logger.warning("commtool notify error (to=%s): %s", to, e)
    return False


async def send_email(to_list: Iterable[str], subject: str, body: str) -> bool:
    """用 mailtool 寄信（週報）。失敗不 raise。"""
    to_str = ",".join(to_list)
    try:
        env = {**os.environ, "MAIL_ACCOUNT": os.environ.get("MAIL_ACCOUNT", "eric")}
        proc = await asyncio.create_subprocess_exec(
            "mailtool.pyz", "send",
            "--to", to_str,
            "--subject", subject,
            "--body", body,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode != 0:
            logger.warning("mailtool send fail (to=%s rc=%d): %s",
                           to_str, proc.returncode, stderr.decode(errors="replace"))
            return False
        logger.info("mailtool email sent to %s", to_str)
        return True
    except FileNotFoundError:
        logger.warning("mailtool.pyz not found, skipping email (to=%s)", to_str)
    except asyncio.TimeoutError:
        logger.warning("mailtool send timeout (to=%s)", to_str)
    except Exception as e:
        logger.warning("mailtool send error (to=%s): %s", to_str, e)
    return False


# ---------------------------------------------------------------------------
# 高階事件封裝
# ---------------------------------------------------------------------------

CTO_TARGET = os.environ.get("MODELHUB_CTO_USERNAME", "cto@hurricanecore.internal")

# M12 N4: brain-cloud webhook dispatch 設定
BRAIN_CLOUD_URL = os.environ.get("BRAIN_CLOUD_URL", "http://brain-cloud-dev:8932")
BRAIN_API_KEY = os.environ.get("BRAIN_API_KEY", "")


async def _dispatch_brain_webhook(
    event_type: str,
    submission,
    model_version: Optional[str] = None,
    map50: Optional[float] = None,
) -> None:
    """
    M12 N4: 呼叫 brain-cloud POST /internal/dispatch-webhook，
    推送事件給所有訂閱方（dagongzai 等外部服務）。
    靜默失敗，不影響主流程。
    """
    if not BRAIN_API_KEY:
        logger.debug("BRAIN_API_KEY not set, skip webhook dispatch (event=%s)", event_type)
        return

    try:
        import httpx

        payload = {
            "event_type": event_type,
            "req_no": submission.req_no,
        }
        if model_version is not None:
            payload["model_version"] = model_version
        if map50 is not None:
            payload["map50"] = map50

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{BRAIN_CLOUD_URL}/internal/dispatch-webhook",
                json=payload,
                headers={"X-Internal-Key": BRAIN_API_KEY},
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info(
                    "notify_event %s: brain webhook dispatch done "
                    "(req=%s subs=%d success=%d fail=%d)",
                    event_type, submission.req_no,
                    data.get("subscriptions_found", 0),
                    data.get("success_count", 0),
                    data.get("fail_count", 0),
                )
            else:
                logger.warning(
                    "notify_event %s: brain webhook dispatch failed (req=%s status=%d)",
                    event_type, submission.req_no, resp.status_code,
                )
    except Exception as e:
        logger.warning(
            "notify_event %s: brain webhook dispatch error (req=%s): %s",
            event_type, getattr(submission, "req_no", "?"), e,
        )


async def notify_event(
    event: str,
    submission,
    actor: Optional[str] = None,
    note: Optional[str] = None,
) -> None:
    """依事件類型發對應通知，靜默失敗。"""
    try:
        submitter = submission.submitter
        req_no = submission.req_no
        title = submission.req_name or submission.product

        if event in ("submit", "resubmit"):
            prefix = "新需求單待審" if event == "submit" else "補件 resubmit 待重新審核"
            extra = ""
            if event == "resubmit":
                extra = f"（第 {submission.resubmit_count} 次送審）"
            msg = f"[ModelHub] {prefix}：{req_no} — {title}{extra}"
            if note:
                msg += f"\n補充：{note}"
            await notify(CTO_TARGET, msg)

        elif event == "approve":
            if submitter:
                msg = f"[ModelHub] 需求單 {req_no} 已核准，可進入訓練階段。"
                if note:
                    msg += f"\n審核意見：{note}"
                await notify(submitter, msg)

        elif event == "reject":
            if submitter:
                msg = (
                    f"[ModelHub] 需求單 {req_no} 已退件，請補件後 resubmit。\n"
                    f"具體缺失與說明請到系統詳情頁查看。"
                )
                if note:
                    msg += f"\n說明：{note}"
                await notify(submitter, msg)

        elif event == "training_complete":
            recipients = {CTO_TARGET}
            if submitter:
                recipients.add(submitter)
            for r in recipients:
                await notify(r, f"[ModelHub] 需求單 {req_no} 訓練完成，模型版本已入庫等候驗收。")
            # M12 N4: 通知外部訂閱方（dagongzai 等）
            map50_val = note  # note 欄位可傳 map50（float 或 None），透過 caller 注入
            await _dispatch_brain_webhook(
                event_type="training_complete",
                submission=submission,
                model_version=getattr(submission, "latest_model_version", None),
                map50=float(map50_val) if map50_val and isinstance(map50_val, (int, float, str)) else None,
            )

        elif event == "training_failed":
            recipients = {CTO_TARGET}
            if submitter:
                recipients.add(submitter)
            for r in recipients:
                await notify(r, f"[ModelHub] 需求單 {req_no} 訓練失敗，請到系統查看 log。")
            # M12 N4: 通知外部訂閱方
            await _dispatch_brain_webhook(
                event_type="training_failed",
                submission=submission,
            )

        elif event == "training_overtime":
            msg = f"[ModelHub] 需求單 {req_no} 訓練已超過 24 小時未完成，可能卡住，請確認。"
            recipients = {CTO_TARGET}
            if submitter:
                recipients.add(submitter)
            for r in recipients:
                await notify(r, msg)

        elif event == "accept":
            if submitter:
                await notify(submitter, f"[ModelHub] 需求單 {req_no} 模型驗收通過。")

        elif event == "fail":
            if submitter:
                await notify(submitter, f"[ModelHub] 需求單 {req_no} 模型驗收未通過。")

        elif event == "retrain":
            await notify(CTO_TARGET, f"[ModelHub] 需求單 {req_no} 已排入重新訓練。")

    except Exception as e:
        logger.warning("notify_event failed (event=%s req=%s): %s",
                       event, getattr(submission, "req_no", "?"), e)
