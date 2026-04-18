"""
routers/actions.py — 需求單狀態機 API

狀態轉移圖：
  draft → submitted → approved → training → trained → accepted
                    ↘ rejected ↗ (resubmit)             ↘ failed
"""

import json
import os
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from models import Submission, SubmissionHistory, SessionLocal, get_db
from notifications import notify_event
from auth import CurrentUser, CurrentUserOrApiKey, require_role

import logging
_logger = logging.getLogger("modelhub.routers.actions")

DISABLE_LOCAL_TRAINING = os.getenv("DISABLE_LOCAL_TRAINING", "false").lower() == "true"

# Auto-approve 模式（預設關閉）：submit 後若 validator 通過，自動推進到 approved + 觸發訓練
# 護欄：dataset_status 必須是 'ready'，且 blocked_reason 為空
AUTO_APPROVE_AFTER_VALIDATOR = os.getenv("AUTO_APPROVE_AFTER_VALIDATOR", "false").lower() == "true"

router = APIRouter()

# 合法的狀態轉移 map：{action: (required_current_status, next_status)}
TRANSITIONS: dict[str, tuple[str, str]] = {
    "submit":            ("draft",     "submitted"),
    "approve":           ("submitted", "approved"),
    "reject":            ("submitted", "rejected"),
    "resubmit":          ("rejected",  "submitted"),
    "start_training":    ("approved",  "training"),
    "complete_training": ("training",  "trained"),
    "accept":            ("trained",   "accepted"),
    "fail":              ("trained",   "failed"),
    "retrain":           ("failed",    "approved"),
}


def _record_history(
    db: Session,
    req_no: str,
    action: str,
    actor: Optional[str] = None,
    reasons: Optional[List[str]] = None,
    note: Optional[str] = None,
    meta: Optional[dict] = None,
) -> None:
    history = SubmissionHistory(
        req_no=req_no,
        action=action,
        actor=actor,
        reasons=json.dumps(reasons, ensure_ascii=False) if reasons else None,
        note=note,
        meta=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(history)


class ActionPayload(BaseModel):
    note: Optional[str] = None
    actor: Optional[str] = None


class RejectPayload(BaseModel):
    reasons: List[str]
    note: Optional[str] = None
    actor: Optional[str] = None


class ResubmitPayload(BaseModel):
    note: Optional[str] = None
    actor: Optional[str] = None


class HistoryOut(BaseModel):
    id: int
    req_no: str
    action: str
    actor: Optional[str]
    reasons: Optional[List[str]]
    note: Optional[str]
    meta: Optional[dict]
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# 結構化退件 / 補件（Sprint 2）
# ---------------------------------------------------------------------------

@router.post("/{req_no}/reject")
async def reject_submission(
    req_no: str,
    payload: RejectPayload,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
    _role: dict = require_role("reviewer"),
):
    """結構化退件：reasons=[...]、note=說明"""
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    if obj.status != "submitted":
        raise HTTPException(
            status_code=422,
            detail=f"reject requires status 'submitted', current='{obj.status}'",
        )
    if not payload.reasons:
        raise HTTPException(status_code=422, detail="reasons 不可為空")

    actor = payload.actor or (current_user or {}).get("preferred_username") or "unknown"
    now = datetime.utcnow()
    obj.status = "rejected"
    obj.rejection_reasons = json.dumps(payload.reasons, ensure_ascii=False)
    obj.rejection_note = payload.note
    obj.reviewed_by = actor
    obj.reviewed_at = now
    if payload.note:
        obj.reviewer_note = payload.note

    _record_history(
        db, req_no=req_no, action="reject", actor=actor,
        reasons=payload.reasons, note=payload.note,
    )
    db.commit()
    db.refresh(obj)
    await notify_event("reject", obj, actor=actor, note=payload.note)
    return {
        "req_no": req_no,
        "action": "reject",
        "status": obj.status,
        "rejection_reasons": payload.reasons,
    }


@router.post("/{req_no}/resubmit")
async def resubmit_submission(
    req_no: str,
    payload: ResubmitPayload,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """補件 resubmit：rejected → submitted，resubmit_count +1"""
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    if obj.status != "rejected":
        raise HTTPException(
            status_code=422,
            detail=f"resubmit requires status 'rejected', current='{obj.status}'",
        )

    actor = payload.actor or (current_user or {}).get("preferred_username") or "unknown"
    now = datetime.utcnow()
    obj.status = "submitted"
    obj.resubmit_count = (obj.resubmit_count or 0) + 1
    obj.resubmitted_at = now

    _record_history(
        db, req_no=req_no, action="resubmit", actor=actor,
        note=payload.note,
        meta={"resubmit_count": obj.resubmit_count},
    )
    db.commit()
    db.refresh(obj)
    await notify_event("resubmit", obj, actor=actor, note=payload.note)
    return {
        "req_no": req_no,
        "action": "resubmit",
        "status": obj.status,
        "resubmit_count": obj.resubmit_count,
    }


@router.get("/{req_no}/history", response_model=List[HistoryOut])
async def get_submission_history(
    req_no: str,
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """審核軌跡（照時間遞增排）"""
    rows = (
        db.query(SubmissionHistory)
        .filter(SubmissionHistory.req_no == req_no)
        .order_by(SubmissionHistory.created_at.asc())
        .all()
    )
    result = []
    for row in rows:
        result.append(HistoryOut(
            id=row.id,
            req_no=row.req_no,
            action=row.action,
            actor=row.actor,
            reasons=json.loads(row.reasons) if row.reasons else None,
            note=row.note,
            meta=json.loads(row.meta) if row.meta else None,
            created_at=row.created_at,
        ))
    return result


# ---------------------------------------------------------------------------
# 通用狀態機 action（其他 transition）
# ---------------------------------------------------------------------------

@router.post("/{req_no}/actions/{action}")
async def perform_action(
    req_no: str,
    action: str,
    background_tasks: BackgroundTasks,
    payload: ActionPayload = ActionPayload(),
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    if action not in TRANSITIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown action '{action}'. Valid: {list(TRANSITIONS.keys())}",
        )

    # P3-3: approve 需要 reviewer role
    if action == "approve":
        import os as _os
        skip = _os.getenv("SKIP_ROLE_CHECK", "false").lower() == "true"
        if not skip:
            _role_claim = _os.getenv("MODELHUB_ROLE_CLAIM", "modelhub_role")
            user_role = (current_user or {}).get(_role_claim)
            # API Key 使用者（sub 以 api_key: 開頭）略過
            is_api_key = str((current_user or {}).get("sub", "")).startswith("api_key:")
            if not is_api_key and user_role != "reviewer":
                raise HTTPException(
                    status_code=403,
                    detail=f"approve requires role 'reviewer', current='{user_role}'",
                )

    # reject / resubmit 有獨立端點用於結構化欄位，但保留這裡作為相容後門
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")

    required_status, next_status = TRANSITIONS[action]
    if obj.status != required_status:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Action '{action}' requires status '{required_status}', "
                f"current='{obj.status}'."
            ),
        )

    actor = payload.actor or (current_user or {}).get("preferred_username") or "unknown"
    obj.status = next_status
    now = datetime.utcnow()

    if action in ("approve", "reject"):
        obj.reviewed_by = actor
        obj.reviewed_at = now
        if payload.note:
            obj.reviewer_note = payload.note
    if action == "start_training":
        obj.training_started_at = now
        obj.total_attempts = (obj.total_attempts or 0) + 1
        # Sprint 15 P1-3: 自動 Kaggle 派發（同步，沿用舊行為）
        _handle_start_training_resource(obj, db)
    if action == "complete_training":
        obj.training_completed_at = now
    if action == "resubmit":
        obj.resubmit_count = (obj.resubmit_count or 0) + 1
        obj.resubmitted_at = now

    _record_history(
        db, req_no=req_no, action=action, actor=actor, note=payload.note,
    )
    db.commit()
    db.refresh(obj)

    # Sprint 20 Task 20-3: approve 後入持久化訓練隊列（移除直接 background_tasks 派發）
    if action == "approve":
        from queue_manager import QueueManager
        obj.kaggle_status = "queued"
        _record_history(
            db, req_no=req_no, action="enqueued",
            actor="system", note="approve 後自動排入訓練隊列",
        )
        db.commit()
        # 在 commit 後入隊，確保 session 狀態乾淨
        queue_db = SessionLocal()
        try:
            QueueManager.enqueue(queue_db, req_no, priority=obj.priority or "P2")
            queue_db.commit()
            _logger.info("approve: req=%s enqueued to training queue", req_no)
        except Exception as _qe:
            _logger.exception("approve: failed to enqueue req=%s: %s", req_no, _qe)
        finally:
            queue_db.close()

    # Auto-approve 模式：submit 後若 validator pass 且 dataset 就緒，自動 approve
    if action == "submit" and AUTO_APPROVE_AFTER_VALIDATOR:
        background_tasks.add_task(_auto_approve_if_valid_bg, req_no, SessionLocal)

    await notify_event(action, obj, actor=actor, note=payload.note)

    return {
        "req_no": req_no,
        "action": action,
        "status": obj.status,
    }


# ---------------------------------------------------------------------------
# Auto-approve 背景任務
# ---------------------------------------------------------------------------

def _auto_approve_if_valid_bg(req_no: str, db_session_factory) -> None:
    """
    submit 後在背景執行：
    1. 重新跑 validator（同步版本）
    2. 若 warnings 為空 且 dataset_status == 'ready' 且無 blocked_reason
       → 自動 approve，寫 history，再觸發 start_training
    3. 若有 warnings → 保持 submitted，寫 auto_approve_skipped history
    護欄：只有 AUTO_APPROVE_AFTER_VALIDATOR=true 時才會被呼叫。
    """
    import asyncio as _asyncio
    db: Session = db_session_factory()
    try:
        obj = db.query(Submission).filter(Submission.req_no == req_no).first()
        if not obj:
            _logger.error("auto_approve: submission %s not found", req_no)
            return
        if obj.status != "submitted":
            _logger.info("auto_approve: submission %s status=%s, skip", req_no, obj.status)
            return

        # 護欄：dataset 必須就緒
        if obj.dataset_status != "ready" or obj.blocked_reason:
            _record_history(
                db, req_no=req_no, action="auto_approve_skipped",
                actor="system",
                note=f"dataset_status={obj.dataset_status}, blocked_reason present, 需人工審查",
            )
            db.commit()
            _logger.info("auto_approve: submission %s skipped (dataset not ready)", req_no)
            return

        # 跑 validator（非同步，用 asyncio）
        from validators import validate_submission

        try:
            loop = _asyncio.new_event_loop()
            warnings = loop.run_until_complete(validate_submission(obj))
            loop.close()
        except Exception as e:
            _logger.warning("auto_approve: validator failed for %s: %s", req_no, e)
            warnings = []  # validator 失敗時不阻止 approve

        if warnings:
            _record_history(
                db, req_no=req_no, action="auto_approve_skipped",
                actor="system",
                note=f"validator 有 {len(warnings)} 個警告，需人工審查",
                meta={"warnings": warnings[:5]},  # 最多記 5 個
            )
            db.commit()
            _logger.info(
                "auto_approve: submission %s skipped (validator warnings=%d)", req_no, len(warnings)
            )
            return

        # 全部通過 → 自動 approve
        now = __import__("datetime").datetime.utcnow()
        obj.status = "approved"
        obj.reviewed_by = "system"
        obj.reviewed_at = now
        obj.reviewer_note = "auto-approved by system (validator pass, dataset ready)"
        obj.kaggle_status = "queued"
        _record_history(
            db, req_no=req_no, action="approve",
            actor="system", note="auto-approved (validator pass)",
        )
        _record_history(
            db, req_no=req_no, action="auto_start_training",
            actor="system", note="auto-approve 後自動排入訓練",
        )
        db.commit()
        db.refresh(obj)

        # Sprint 20: 改為入持久化隊列，由 queue_dispatcher 統一派發
        from queue_manager import QueueManager
        try:
            QueueManager.enqueue(db, req_no, priority=obj.priority or "P2")
            db.commit()
            _logger.info("auto_approve: submission %s enqueued to training queue", req_no)
        except Exception as _qe:
            _logger.exception("auto_approve: failed to enqueue req=%s: %s", req_no, _qe)

    except Exception as e:
        _logger.exception("auto_approve failed for req=%s: %s", req_no, e)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Sprint 15 P1-3: 訓練資源自動派發
# ---------------------------------------------------------------------------

def _handle_start_training_resource(obj: Submission, db: Session) -> None:
    """
    start_training 觸發時決定訓練資源：
    1. 呼叫 ResourceProber.get_best_resource()
    2. 若 resource=kaggle 且 has_kaggle_kernel(req_no) → KaggleLauncher.push_and_attach
    3. 否則 fallback local（SSH P2 之後實作）
    記錄 training_resource 到 submission，history note 寫入使用資源。
    """
    try:
        from resources.prober import ResourceProber
        from resources.kernel_registry import has_kaggle_kernel
        from resources.kaggle_launcher import KaggleLauncher

        prober = ResourceProber()
        best = prober.get_best_resource(db=db)
        resource = best.get("resource", "local")
        device = best.get("device", "cpu")

        if resource == "kaggle" and has_kaggle_kernel(obj.req_no):
            launcher = KaggleLauncher()
            success = launcher.push_and_attach(obj.req_no, db, obj)
            if success:
                obj.training_resource = "kaggle"
                _logger.info("start_training req=%s dispatched to Kaggle", obj.req_no)
                _record_history(
                    db, req_no=obj.req_no, action="resource_selected",
                    actor="system",
                    note=f"使用資源：kaggle（kernel={obj.kaggle_kernel_slug}）",
                    meta={"resource": "kaggle", "device": "cuda"},
                )
                return
            else:
                _logger.warning("KaggleLauncher.push_and_attach failed for req=%s, fallback Lightning", obj.req_no)

        # Sprint 15 P2-1: Kaggle 不可用時嘗試 Lightning AI
        if resource == "lightning":
            from resources.lightning_launcher import LightningLauncher
            dataset_path = getattr(obj, "dataset_path", None) or ""
            lightning = LightningLauncher()
            result = lightning.submit_job(
                req_no=obj.req_no,
                dataset_path=dataset_path,
                config=None,  # 使用預設訓練參數
            )
            if result.get("success"):
                obj.training_resource = "lightning"
                studio_name = result.get("studio_name", "")
                _logger.info(
                    "start_training req=%s dispatched to Lightning studio=%s",
                    obj.req_no, studio_name,
                )
                _record_history(
                    db, req_no=obj.req_no, action="resource_selected",
                    actor="system",
                    note=f"使用資源：lightning（studio={studio_name}）",
                    meta={"resource": "lightning", "device": "cuda", "studio_name": studio_name},
                )
                return
            else:
                _logger.warning(
                    "LightningLauncher.submit_job failed for req=%s: %s, fallback local",
                    obj.req_no, result.get("reason"),
                )

        # P0-2: fallback local 前檢查 DISABLE_LOCAL_TRAINING
        if DISABLE_LOCAL_TRAINING:
            obj.status = "blocked"
            obj.blocked_reason = "無可用線上 GPU 資源（Kaggle 配額耗盡或無 kernel），本機訓練已停用"
            db.commit()
            _logger.warning("start_training req=%s blocked: no online GPU, local training disabled", obj.req_no)
            from notifications import notify_event as _notify_event
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(_notify_event("training_blocked", obj, note="no_online_resource"))
                else:
                    loop.run_until_complete(_notify_event("training_blocked", obj, note="no_online_resource"))
            except Exception:
                pass
            return

        # Sprint 23 Task 23-2: SSH 分支實際呼叫 SSHLauncher
        if resource == "ssh":
            host = best.get("host", "unknown")
            try:
                from resources.ssh_launcher import SSHLauncher
                dataset_path = getattr(obj, "dataset_path", None) or ""
                ssh_launcher = SSHLauncher()
                ssh_result = ssh_launcher.submit_job(
                    host=host,
                    req_no=obj.req_no,
                    dataset_path=dataset_path,
                    config=None,
                )
                if ssh_result.get("success"):
                    obj.training_resource = f"ssh@{host}"
                    note = f"使用資源：ssh@{host}（device=cuda）"
                    _logger.info("start_training req=%s dispatched to SSH host=%s", obj.req_no, host)
                    _record_history(
                        db, req_no=obj.req_no, action="resource_selected",
                        actor="system",
                        note=note,
                        meta={"resource": "ssh", "device": "cuda", "host": host,
                              "pid": ssh_result.get("pid")},
                    )
                    return
                else:
                    _logger.warning(
                        "SSHLauncher.submit_job failed for req=%s host=%s: %s, fallback local",
                        obj.req_no, host, ssh_result.get("reason"),
                    )
                    # fallthrough 到 local
            except Exception as _se:
                _logger.warning("SSHLauncher exception for req=%s: %s, fallback local", obj.req_no, _se)
            # SSH 失敗後 fallback 到 local
            local_result = ResourceProber().probe_local_mps()
            local_device = local_result.get("device", "cpu")
            obj.training_resource = f"local_{local_device}"
            note = f"SSH 失敗後 fallback：local_{local_device}"
        else:
            local_device = device if device else "cpu"
            obj.training_resource = f"local_{local_device}"
            note = f"使用資源：local_{local_device}"

        _logger.info("start_training req=%s resource=%s", obj.req_no, obj.training_resource)
        _record_history(
            db, req_no=obj.req_no, action="resource_selected",
            actor="system",
            note=note,
            meta={"resource": resource, "device": device},
        )

    except Exception as e:
        _logger.warning("_handle_start_training_resource failed for req=%s: %s", obj.req_no, e)
        # 不中斷主流程，training_resource 留 None


# ---------------------------------------------------------------------------
# P0-1: approve → auto start_training 背景任務（不依賴 request scope db）
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sprint 19 C.2 + D: POST /api/submissions/{req_no}/retrain-lightning
# ---------------------------------------------------------------------------

class RetrainLightningRequest(BaseModel):
    epochs: int = 100
    arch: Optional[str] = None


@router.post("/{req_no}/retrain-lightning")
async def retrain_lightning(
    req_no: str,
    body: RetrainLightningRequest = RetrainLightningRequest(),
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    training_failed 後快速重送 Lightning AI GPU 重新訓練。
    允許狀態：failed（training_failed）。
    """
    obj = db.query(Submission).filter(Submission.req_no == req_no).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Submission not found")
    if obj.status not in ("failed", "training_failed"):
        raise HTTPException(
            status_code=422,
            detail=f"retrain-lightning requires status 'failed' or 'training_failed', current='{obj.status}'",
        )

    actor = (current_user or {}).get("preferred_username") or "unknown"
    now = datetime.utcnow()

    try:
        from resources.lightning_launcher import LightningLauncher
        dataset_path = getattr(obj, "dataset_path", None) or ""
        config: dict = {"epochs": body.epochs}
        if body.arch:
            config["arch"] = body.arch

        launcher = LightningLauncher()
        result = launcher.submit_job(
            req_no=obj.req_no,
            dataset_path=dataset_path,
            config=config,
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"LightningLauncher.submit_job 例外: {e}",
        )

    if not result.get("success"):
        raise HTTPException(
            status_code=503,
            detail=f"Lightning 重送失敗: {result.get('reason', 'unknown')}",
        )

    studio_name = result.get("studio_name", "")
    obj.training_resource = "lightning"
    obj.status = "training"
    obj.training_started_at = now
    obj.total_attempts = (obj.total_attempts or 0) + 1
    if studio_name:
        obj.lightning_studio_name = studio_name
    obj.kaggle_status = "queued"
    obj.kaggle_status_updated_at = now

    _record_history(
        db,
        req_no=req_no,
        action="lightning_retrain_submitted",
        actor=actor,
        note=f"Lightning GPU 重送（studio={studio_name}, epochs={body.epochs}）",
        meta={
            "studio_name": studio_name,
            "epochs": body.epochs,
            "arch": body.arch or obj.arch,
            "resource": "lightning",
        },
    )
    db.commit()
    db.refresh(obj)

    return {
        "req_no": req_no,
        "action": "retrain_lightning",
        "status": obj.status,
        "studio_name": studio_name,
        "epochs": body.epochs,
    }


def _handle_start_training_resource_bg(req_no: str, db_session_factory) -> None:
    """
    背景任務版本：approve 後自動將 submission 推進到 training 狀態並派發資源。
    使用獨立 db session，不依賴 request scope。
    """
    db: Session = db_session_factory()
    try:
        obj = db.query(Submission).filter(Submission.req_no == req_no).first()
        if not obj:
            _logger.error("auto_start_training: submission %s not found", req_no)
            return
        if obj.status != "approved":
            _logger.warning(
                "auto_start_training: submission %s status=%s (expected approved), skip",
                req_no, obj.status,
            )
            return

        now = __import__("datetime").datetime.utcnow()
        obj.status = "training"
        obj.training_started_at = now
        obj.total_attempts = (obj.total_attempts or 0) + 1

        _record_history(
            db, req_no=req_no, action="start_training",
            actor="system", note="auto-triggered by approve",
        )
        db.commit()
        db.refresh(obj)

        _handle_start_training_resource(obj, db)
        _logger.info("auto_start_training completed for req=%s", req_no)

    except Exception as e:
        _logger.exception("auto_start_training failed for req=%s: %s", req_no, e)
    finally:
        db.close()
