"""
週報（Sprint 5）：每週一 09:00 Asia/Taipei 寄給 eric + anderson

P3-2: 改呼叫 Anemone mail endpoint，mailtool subprocess 作 fallback。
"""
import logging
import os
from datetime import datetime, timedelta

import httpx

from models import SessionLocal, Submission
from notifications import send_email

logger = logging.getLogger("modelhub.weekly_report")

DEFAULT_RECIPIENTS = [
    "eric.wang@hurricanesoft.com.tw",
    "anderson@hurricanecore.internal",
]

ANEMONE_URL = os.getenv("ANEMONE_API_URL", "http://anemone-api-dev:8920").rstrip("/")
ANEMONE_TOKEN = os.getenv("ANEMONE_API_TOKEN", "")


async def _send_via_anemone(to_list: list[str], subject: str, body: str) -> bool:
    """透過 Anemone mail endpoint 寄信。未設 token 時靜默略過，回傳 False。"""
    if not ANEMONE_TOKEN:
        return False
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{ANEMONE_URL}/v1/mail/send",
                headers={
                    "Authorization": f"Bearer {ANEMONE_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "to": to_list,
                    "subject": subject,
                    "body": body,
                },
            )
            if resp.status_code == 200:
                logger.info("weekly report sent via Anemone to %s", to_list)
                return True
            logger.warning("Anemone mail non-200 (%d): %s", resp.status_code, resp.text[:200])
            return False
    except Exception as e:
        logger.warning("Anemone mail error: %s", e)
        return False


def _count_by_status_since(rows, status: str, since: datetime) -> int:
    return sum(1 for r in rows if r.status == status and r.created_at >= since)


async def send_weekly_report():
    now = datetime.utcnow()
    since = now - timedelta(days=7)
    db = SessionLocal()
    try:
        all_rows = db.query(Submission).all()

        # 本週活動統計（使用正確時間欄位）
        new_subs = [r for r in all_rows if r.created_at >= since]
        total_new = len(new_subs)
        approved = sum(1 for r in all_rows
                       if r.status == "approved" and r.reviewed_at and r.reviewed_at >= since)
        rejected = sum(1 for r in all_rows
                       if r.status == "rejected" and r.reviewed_at and r.reviewed_at >= since)
        trained = sum(1 for r in all_rows
                      if r.status == "trained" and r.training_completed_at and r.training_completed_at >= since)

        # 訓練中
        training_rows = [r for r in all_rows if r.status == "training"]
        training_lines = [
            f"- {r.req_no} / {r.req_name or r.product} — kernel={r.kaggle_status or 'n/a'}, "
            f"started={r.training_started_at.strftime('%Y-%m-%d %H:%M') if r.training_started_at else '-'}"
            for r in training_rows
        ]

        # 超時（submitted > 7 天未審）
        overdue_rows = [
            r for r in all_rows
            if r.status == "submitted" and r.created_at < since
        ]
        overdue_lines = [
            f"- {r.req_no} / {r.req_name or r.product} — 已等 {(now - r.created_at).days} 天"
            for r in overdue_rows
        ]

        # GPU 時數（所有 training_completed_at >= since 的工單累計）
        trained_rows = [r for r in all_rows
                        if r.training_completed_at and r.training_completed_at >= since]
        total_gpu_seconds = sum(r.gpu_seconds or 0 for r in trained_rows)
        total_cost = sum(r.estimated_cost_usd or 0 for r in trained_rows)

        body = f"""ModelHub 週報 — {now.strftime('%Y-%m-%d')}

本週活動（近 7 天）:
- 新建需求單：{total_new}
- 核准：{approved}
- 退件：{rejected}
- 訓練完成：{trained}

訓練中（{len(training_rows)}）:
{chr(10).join(training_lines) if training_lines else '(無)'}

超時工單（submitted > 7 天）：
{chr(10).join(overdue_lines) if overdue_lines else '(無)'}

累計 GPU 時數：{total_gpu_seconds // 3600} 小時 {(total_gpu_seconds % 3600) // 60} 分鐘
估算成本：USD ${total_cost:.2f}

—— ModelHub 自動產生
"""
        subject = f"[ModelHub] 週報 {now.strftime('%Y-%m-%d')}"
        recipients_str = os.environ.get("MODELHUB_REPORT_RECIPIENTS", ",".join(DEFAULT_RECIPIENTS))
        recipients = recipients_str.split(",")
        # P3-2: 優先走 Anemone mail endpoint，失敗再 fallback 到 mailtool subprocess
        sent = await _send_via_anemone(recipients, subject, body)
        if not sent:
            await send_email(recipients, subject, body)
    except Exception as e:
        logger.warning("weekly report failed: %s", e)
    finally:
        db.close()
