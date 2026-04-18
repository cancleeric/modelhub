"""
backend/resources/prober.py — 訓練資源探測模組（Sprint 15 P1-1, P1-4, P2-2）

核心原則：訓練前先找免費 GPU，順序：Kaggle → 內網 GPU → 本機 MPS
"""

import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("modelhub.resources.prober")

# 從 env 讀內網 GPU 主機清單
# 格式：TRAINING_SSH_HOSTS=user@192.168.50.83,user@192.168.0.70
_SSH_HOSTS_ENV = os.environ.get("TRAINING_SSH_HOSTS", "")


def _parse_ssh_hosts() -> list[str]:
    if not _SSH_HOSTS_ENV.strip():
        return []
    return [h.strip() for h in _SSH_HOSTS_ENV.split(",") if h.strip()]


class KaggleQuotaTracker:
    """Sprint 15 P1-4: 追蹤 Kaggle 免費配額用量"""

    WEEKLY_LIMIT_HOURS = 30

    def get_used_hours_this_week(self, db) -> float:
        """查本週所有 Kaggle 訓練的 gpu_seconds 總和 / 3600"""
        from models import Submission
        # 本週一 00:00:00 UTC
        today = datetime.utcnow()
        monday = today - timedelta(days=today.weekday())
        week_start = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        rows = (
            db.query(Submission)
            .filter(
                Submission.training_resource == "kaggle",
                Submission.training_started_at >= week_start,
                Submission.gpu_seconds.isnot(None),
            )
            .all()
        )
        total_seconds = sum(r.gpu_seconds for r in rows if r.gpu_seconds)
        return total_seconds / 3600.0

    def get_remaining_hours(self, db) -> float:
        """本週剩餘配額（小時）"""
        used = self.get_used_hours_this_week(db)
        return max(0.0, self.WEEKLY_LIMIT_HOURS - used)

    def is_quota_available(self, db, estimated_hours: float = 2.0) -> bool:
        """used + estimated <= WEEKLY_LIMIT_HOURS"""
        used = self.get_used_hours_this_week(db)
        return (used + estimated_hours) <= self.WEEKLY_LIMIT_HOURS


class ResourceProber:
    """Sprint 15 P1-1: 訓練資源探測器"""

    def probe_kaggle(self, db=None) -> dict:
        """
        檢查 Kaggle 是否可用：
        1. KAGGLE_USERNAME + KAGGLE_KEY env 是否已設
        2. kaggle CLI 是否可達（kaggle kernels list --mine --csv，timeout 5s）
        3. 若有 db，整合 quota check
        """
        username = os.environ.get("KAGGLE_USERNAME", "")
        key = os.environ.get("KAGGLE_KEY", "")

        if not username or not key:
            return {"available": False, "reason": "KAGGLE_USERNAME or KAGGLE_KEY not set"}

        if not shutil.which("kaggle"):
            return {"available": False, "reason": "kaggle CLI not in PATH"}

        try:
            result = subprocess.run(
                ["kaggle", "kernels", "list", "--mine", "--csv"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return {
                    "available": False,
                    "reason": f"kaggle CLI error: {result.stderr.strip()[:200]}",
                }
        except subprocess.TimeoutExpired:
            return {"available": False, "reason": "kaggle CLI timeout (5s)"}
        except Exception as e:
            return {"available": False, "reason": str(e)}

        # quota check
        if db is not None:
            try:
                tracker = KaggleQuotaTracker()
                if not tracker.is_quota_available(db):
                    used = tracker.get_used_hours_this_week(db)
                    return {
                        "available": False,
                        "reason": f"Kaggle weekly quota exhausted ({used:.1f}h / {KaggleQuotaTracker.WEEKLY_LIMIT_HOURS}h used)",
                    }
            except Exception as e:
                logger.warning("Kaggle quota check failed: %s", e)

        return {"available": True, "reason": "ok"}

    def probe_ssh_host(self, host: str) -> dict:
        """
        SSH ping host，執行 nvidia-smi 查 GPU 狀態。
        Sprint 15 P2-2
        """
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", "ConnectTimeout=3",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "BatchMode=yes",
                    host,
                    "nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return {
                    "available": False,
                    "reason": f"nvidia-smi failed (rc={result.returncode}): {result.stderr.strip()[:200]}",
                    "host": host,
                }

            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if not lines:
                return {"available": False, "reason": "nvidia-smi no output", "host": host}

            gpus = []
            for line in lines:
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    gpus.append({
                        "util": int(parts[0].strip()),
                        "free_mb": int(parts[1].strip()),
                    })
                except ValueError:
                    continue

            if not gpus:
                return {"available": False, "reason": "nvidia-smi parse failed", "host": host}

            available = any(g["util"] < 80 and g["free_mb"] > 4000 for g in gpus)
            return {
                "available": available,
                "gpu_count": len(gpus),
                "free_memory_mb": max(g["free_mb"] for g in gpus),
                "gpus": gpus,
                "host": host,
            }

        except subprocess.TimeoutExpired:
            return {"available": False, "reason": "SSH timeout (5s)", "host": host}
        except Exception as e:
            return {"available": False, "reason": str(e), "host": host}

    def probe_lightning(self) -> dict:
        """
        P2-1: 檢查 Lightning AI 是否可用。
        若 LIGHTNING_API_KEY 未設，直接回傳不可用。
        若已設，嘗試執行 lightning status 確認連線。
        """
        api_key = os.environ.get("LIGHTNING_API_KEY", "")
        if not api_key:
            return {"available": False, "reason": "LIGHTNING_API_KEY not set"}

        # 嘗試 lightning CLI ping（若有安裝）
        lightning_cli = shutil.which("lightning")
        if lightning_cli:
            try:
                result = subprocess.run(
                    ["lightning", "status"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return {"available": True, "reason": "lightning CLI ok"}
                return {
                    "available": False,
                    "reason": f"lightning CLI error: {result.stderr.strip()[:200]}",
                }
            except subprocess.TimeoutExpired:
                return {"available": False, "reason": "lightning CLI timeout (5s)"}
            except Exception as e:
                return {"available": False, "reason": str(e)}

        # CLI 不在 PATH，但 API Key 已設 — 視為可用（SDK 模式）
        return {"available": True, "reason": "LIGHTNING_API_KEY set (CLI not in PATH, SDK mode)"}

    def probe_local_mps(self) -> dict:
        """檢查本機 MPS（Apple Silicon）或 CPU fallback"""
        try:
            import torch
            if torch.backends.mps.is_available():
                return {"available": True, "device": "mps"}
            return {"available": True, "device": "cpu"}
        except ImportError:
            return {"available": True, "device": "cpu"}

    def get_best_resource(self, submission: Optional[dict] = None, db=None) -> dict:
        """
        P2-1: 按優先序探測，回傳第一個可用資源：
        Kaggle → Lightning → SSH → Local
        """
        # 1. Kaggle
        kaggle_result = self.probe_kaggle(db=db)
        if kaggle_result["available"]:
            return {
                "resource": "kaggle",
                "device": "cuda",
                "details": kaggle_result,
            }
        logger.info("Kaggle not available: %s", kaggle_result.get("reason"))

        # 2. Lightning AI
        lightning_result = self.probe_lightning()
        if lightning_result["available"]:
            return {
                "resource": "lightning",
                "device": "cuda",
                "details": lightning_result,
            }
        logger.info("Lightning not available: %s", lightning_result.get("reason"))

        # 3. SSH hosts（內網 GPU）
        ssh_hosts = _parse_ssh_hosts()
        for host in ssh_hosts:
            ssh_result = self.probe_ssh_host(host)
            if ssh_result["available"]:
                return {
                    "resource": "ssh",
                    "device": "cuda",
                    "host": host,
                    "details": ssh_result,
                }
            logger.info("SSH host %s not available: %s", host, ssh_result.get("reason"))

        # 4. Local MPS / CPU
        local_result = self.probe_local_mps()
        return {
            "resource": "local",
            "device": local_result["device"],
            "details": local_result,
        }
