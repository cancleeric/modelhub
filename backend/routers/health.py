"""
routers/health.py — 資源健康查詢（Sprint 16 P2-2）
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from models import get_db
from auth import CurrentUserOrApiKey

router = APIRouter()


@router.get("/resources")
async def get_resources(
    db: Session = Depends(get_db),
    current_user: dict = CurrentUserOrApiKey,
):
    """
    查詢所有訓練資源的可用狀態。
    回傳 Kaggle 配額、本機 MPS、SSH GPU hosts。
    """
    from resources.prober import ResourceProber, KaggleQuotaTracker
    import os

    prober = ResourceProber()

    # Kaggle
    kaggle_result = prober.probe_kaggle(db=db)
    quota_used = 0.0
    quota_remaining = 30.0
    try:
        tracker = KaggleQuotaTracker()
        quota_used = tracker.get_used_hours_this_week(db)
        quota_remaining = max(0.0, KaggleQuotaTracker.WEEKLY_LIMIT_HOURS - quota_used)
    except Exception:
        pass

    kaggle_info = {
        "available": kaggle_result.get("available", False),
        "quota_used_hours": round(quota_used, 2),
        "quota_remaining": round(quota_remaining, 2),
        "reason": kaggle_result.get("reason"),
    }

    # Local MPS
    local_mps = prober.probe_local_mps()

    # SSH hosts
    ssh_hosts_env = os.environ.get("TRAINING_SSH_HOSTS", "")
    ssh_hosts_list = [h.strip() for h in ssh_hosts_env.split(",") if h.strip()]
    ssh_results = []
    for host in ssh_hosts_list:
        r = prober.probe_ssh_host(host)
        ssh_results.append({
            "host": host,
            "available": r.get("available", False),
            "gpus": r.get("gpus", []),
            "free_memory_mb": r.get("free_memory_mb"),
            "reason": r.get("reason"),
        })

    return {
        "kaggle": kaggle_info,
        "local_mps": {"available": local_mps.get("available", False), "device": local_mps.get("device")},
        "ssh_hosts": ssh_results,
    }
