"""
Lightning AI 訓練啟動器（Sprint 15 P2-1）

需要：LIGHTNING_API_KEY env var（從 lightning.ai 取得）
取得方式：登入 https://lightning.ai → Settings → API Keys → Create API Key

安裝：
    pip install lightning-sdk

用法：
    LIGHTNING_API_KEY=<key> python3 -c "
    from resources.lightning_launcher import LightningLauncher
    l = LightningLauncher()
    print(l.is_available())
    "
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("modelhub.lightning.launcher")

LIGHTNING_API_KEY = os.environ.get("LIGHTNING_API_KEY")

# Lightning AI Studio 設定
LIGHTNING_USERNAME = os.environ.get("LIGHTNING_USERNAME", "boardgamegroup")
LIGHTNING_TEAMSPACE = os.environ.get("LIGHTNING_TEAMSPACE", "boardgamegroup")
LIGHTNING_ORG = os.environ.get("LIGHTNING_ORG", LIGHTNING_TEAMSPACE)


class LightningLauncher:
    """
    Lightning AI 雲端 GPU 訓練啟動器。

    目前狀態：框架已建立，實際 job 提交需要 LIGHTNING_API_KEY。
    Kaggle 免費 GPU 為目前主力，Lightning 作為 P2 備選（無限制 GPU 時數方案）。
    """

    def is_available(self) -> bool:
        """檢查 Lightning API Key 是否已設定"""
        return bool(LIGHTNING_API_KEY)

    def submit_job(
        self,
        req_no: str,
        script_path: str,
        dataset_path: str,
        *,
        machine: str = "gpu",
        num_gpus: int = 1,
        studio_name: Optional[str] = None,
    ) -> dict:
        """
        提交訓練 job 到 Lightning AI 雲端。

        Args:
            req_no: 需求編號，如 "MH-2026-006"
            script_path: 訓練腳本本機路徑
            dataset_path: 資料集本機路徑（需先上傳到 Lightning Studio 或 S3）
            machine: GPU 機型（"gpu" / "gpu-fast" / "gpu-fast-multi"）
            num_gpus: GPU 數量
            studio_name: Lightning Studio 名稱（預設使用 req_no）

        Returns:
            {"success": bool, "job_id": str, "studio": str} 或
            {"success": False, "reason": str}
        """
        if not self.is_available():
            return {
                "success": False,
                "reason": "LIGHTNING_API_KEY 未設定。請至 https://lightning.ai → Settings → API Keys 取得。",
            }

        # TODO: 待 API Key 後實作
        # 預計流程：
        # 1. from lightning_sdk import Studio
        # 2. studio = Studio(name=studio_name, teamspace=LIGHTNING_TEAMSPACE, org=LIGHTNING_ORG)
        # 3. studio.start(machine=machine)
        # 4. studio.run(f"python {script_path}")
        # 5. job_id = studio.name
        # 6. studio.stop()（或讓腳本跑完自動 stop）
        raise NotImplementedError(
            "需要 LIGHTNING_API_KEY 才能測試。\n"
            "取得方式：https://lightning.ai → Settings → API Keys → Create API Key\n"
            "設定後：export LIGHTNING_API_KEY=<key>"
        )

    def get_job_status(self, job_id: str) -> dict:
        """
        查詢 Lightning job 狀態。

        Args:
            job_id: submit_job 回傳的 job_id（即 Studio name）

        Returns:
            {"status": str, "progress": float, "logs": str}
        """
        if not self.is_available():
            return {"status": "unavailable", "reason": "LIGHTNING_API_KEY 未設定"}

        # TODO: 待 API Key 後實作
        # from lightning_sdk import Studio
        # studio = Studio(name=job_id, teamspace=LIGHTNING_TEAMSPACE)
        # status = studio.status
        raise NotImplementedError("需要 LIGHTNING_API_KEY")

    def list_jobs(self) -> list[dict]:
        """列出所有 Lightning 訓練 job"""
        if not self.is_available():
            return []

        # TODO: 待 API Key 後實作
        raise NotImplementedError("需要 LIGHTNING_API_KEY")

    def get_api_key_instructions(self) -> str:
        """回傳取得 API Key 的說明"""
        return (
            "Lightning AI API Key 取得步驟：\n"
            "1. 前往 https://lightning.ai\n"
            "2. 登入帳號（或使用 GitHub 登入）\n"
            "3. 右上角 → Settings → API Keys\n"
            "4. 點擊 'Create API Key'\n"
            "5. 複製 key 並設定環境變數：\n"
            "   export LIGHTNING_API_KEY=<your_key>\n"
            "   也可加入 .env 或 GCP Secret Manager（key: modelhub/prod/LIGHTNING_API_KEY）\n"
            "\n"
            "注意：Lightning AI 提供免費 GPU 配額（T4 22hr/月），超量計費。\n"
            "付費方案提供更多配額（A10G、V100 等）。\n"
        )
