"""
Lightning AI 訓練啟動器（Sprint 15 P2-1）

需要：
  LIGHTNING_USER_ID  — Lightning 帳號 UUID（從 hvault get hurricanecore/dev/LIGHTNING_USER_ID）
  LIGHTNING_API_KEY  — Lightning API Key（從 hvault get hurricanecore/dev/LIGHTNING_API_KEY）

安裝：
    pip install lightning-sdk

用法：
    LIGHTNING_USER_ID=<uid> LIGHTNING_API_KEY=<key> python3 -c "
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

# credentials 從 env 讀取；啟動時建議用 hvault run --prefix hurricanecore/dev/ --
LIGHTNING_USER_ID = os.environ.get("LIGHTNING_USER_ID", "")
LIGHTNING_API_KEY = os.environ.get("LIGHTNING_API_KEY", "")

# Lightning AI Studio 設定
LIGHTNING_USERNAME = os.environ.get("LIGHTNING_USERNAME", "cancleeric")
LIGHTNING_TEAMSPACE = os.environ.get("LIGHTNING_TEAMSPACE", "default-teamspace")


class LightningLauncher:
    """
    Lightning AI 雲端 GPU 訓練啟動器。

    資源優先序：Kaggle（主力）→ Lightning（P2 備選）→ SSH → 本機 MPS
    """

    def is_available(self) -> bool:
        """檢查 LIGHTNING_USER_ID 與 LIGHTNING_API_KEY 是否皆已設定"""
        return bool(LIGHTNING_USER_ID and LIGHTNING_API_KEY)

    # ------------------------------------------------------------------
    # submit_job
    # ------------------------------------------------------------------

    def submit_job(
        self,
        req_no: str,
        dataset_path: str,
        config: Optional[dict] = None,
        *,
        script_path: Optional[str] = None,
        machine: str = "T4",
        studio_name: Optional[str] = None,
    ) -> dict:
        """
        建立/啟動 Lightning AI Studio，上傳 dataset，執行訓練腳本。

        Args:
            req_no:       需求編號，如 "MH-2026-008"
            dataset_path: 本機 dataset 目錄路徑（會整個 upload 到 Studio）
            config:       訓練設定 dict（epochs、imgsz 等，傳遞給訓練腳本）
            script_path:  訓練腳本本機路徑（若 None，使用 dataset_path 內的 train.py）
            machine:      GPU 機型（"T4" / "L4" / "A10G" 等，預設 T4 免費）
            studio_name:  Studio 名稱（預設從 req_no 生成）

        Returns:
            {"success": True, "studio_name": str} 或
            {"success": False, "reason": str}
        """
        if not self.is_available():
            return {
                "success": False,
                "reason": (
                    "LIGHTNING_USER_ID 或 LIGHTNING_API_KEY 未設定。"
                    "取得方式：hvault get hurricanecore/dev/LIGHTNING_USER_ID"
                ),
            }

        if not studio_name:
            # req_no 可能含大寫與底線，Lightning Studio 名稱只接受小寫與 dash
            studio_name = req_no.lower().replace("_", "-")

        try:
            from lightning_sdk import Studio, Machine

            _machine = self._resolve_machine(machine)
            logger.info(
                "submit_job req=%s studio=%s machine=%s",
                req_no, studio_name, machine,
            )

            studio = Studio(
                name=studio_name,
                teamspace=LIGHTNING_TEAMSPACE,
                user=LIGHTNING_USERNAME,
                create_ok=True,
            )
            studio.start(machine=_machine)
            logger.info("Studio %s started", studio_name)

            # 上傳 dataset
            dataset_local = Path(dataset_path)
            if dataset_local.is_dir():
                studio.upload_folder(str(dataset_local), "dataset/")
                logger.info("Dataset folder uploaded: %s → dataset/", dataset_local)
            elif dataset_local.is_file():
                studio.upload_file(str(dataset_local), f"dataset/{dataset_local.name}")
                logger.info("Dataset file uploaded: %s → dataset/%s", dataset_local, dataset_local.name)
            else:
                logger.warning("dataset_path %s not found, skipping upload", dataset_path)

            # 上傳訓練腳本（若有指定）
            if script_path and Path(script_path).is_file():
                studio.upload_file(script_path, "train.py")
                logger.info("Training script uploaded: %s → train.py", script_path)

            # 組裝訓練指令
            train_cmd = self._build_train_command(config)

            # 安裝依賴並啟動訓練（背景執行，run_and_detach 讓呼叫端不阻塞）
            studio.run("pip install ultralytics --quiet")
            studio.run_and_detach(train_cmd)
            logger.info("Training started on studio=%s cmd=%s", studio_name, train_cmd)

            return {"success": True, "studio_name": studio_name}

        except Exception as e:
            logger.exception("submit_job failed for req=%s: %s", req_no, e)
            return {"success": False, "reason": str(e)}

    # ------------------------------------------------------------------
    # get_job_status
    # ------------------------------------------------------------------

    def get_job_status(self, studio_name: str) -> str:
        """
        查詢 Studio 執行狀態。

        Returns:
            "running" | "complete" | "error" | "unknown"
        """
        if not self.is_available():
            return "error"

        try:
            from lightning_sdk import Studio, Status

            studio = Studio(
                name=studio_name,
                teamspace=LIGHTNING_TEAMSPACE,
                user=LIGHTNING_USERNAME,
                create_ok=False,
            )
            status = studio.status

            if status == Status.Running:
                return "running"
            elif status in (Status.Stopped, Status.Completed):
                return "complete"
            elif status == Status.Failed:
                return "error"
            elif status == Status.NotCreated:
                return "unknown"
            else:
                # Pending / Stopping → 視為 running（仍在處理中）
                return "running"

        except Exception as e:
            logger.warning("get_job_status failed for studio=%s: %s", studio_name, e)
            return "error"

    # ------------------------------------------------------------------
    # get_job_logs
    # ------------------------------------------------------------------

    def get_job_logs(self, studio_name: str) -> str:
        """
        取得 Studio 訓練 log（讀取 training.log 或 stdout）。

        Returns:
            log 字串；失敗時回傳空字串
        """
        if not self.is_available():
            return ""

        try:
            from lightning_sdk import Studio

            studio = Studio(
                name=studio_name,
                teamspace=LIGHTNING_TEAMSPACE,
                user=LIGHTNING_USERNAME,
                create_ok=False,
            )

            # 嘗試讀取訓練 log 檔（ultralytics 預設輸出到 runs/train/expN/）
            # 先試標準輸出（run 的 return value）
            try:
                log_content = studio.run("cat training.log 2>/dev/null || echo ''")
                if log_content and log_content.strip():
                    return log_content
            except Exception:
                pass

            # fallback：用 run 取 ultralytics results
            try:
                log_content = studio.run(
                    "find runs/train -name 'results.csv' 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo 'No log found'"
                )
                return log_content or ""
            except Exception:
                return ""

        except Exception as e:
            logger.warning("get_job_logs failed for studio=%s: %s", studio_name, e)
            return ""

    # ------------------------------------------------------------------
    # download_model
    # ------------------------------------------------------------------

    def download_model(self, studio_name: str, local_path: str) -> bool:
        """
        從 Studio 下載 best.pt 到 local_path。

        Args:
            studio_name: Studio 名稱
            local_path:  本機目標路徑（含檔名，例如 /app/data/models/mh-2026-008/best.pt）

        Returns:
            True 若成功，False 若失敗
        """
        if not self.is_available():
            return False

        try:
            from lightning_sdk import Studio

            studio = Studio(
                name=studio_name,
                teamspace=LIGHTNING_TEAMSPACE,
                user=LIGHTNING_USERNAME,
                create_ok=False,
            )

            # 找 best.pt 路徑（ultralytics 預設輸出位置）
            remote_best_pt = studio.run(
                "find runs/train -name 'best.pt' 2>/dev/null | head -1"
            ).strip()

            if not remote_best_pt:
                logger.warning("download_model: best.pt not found in studio=%s", studio_name)
                return False

            # 確保本機目錄存在
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            studio.download_file(remote_best_pt, local_path)
            logger.info(
                "download_model: %s:%s → %s", studio_name, remote_best_pt, local_path
            )
            return True

        except Exception as e:
            logger.exception("download_model failed for studio=%s: %s", studio_name, e)
            return False

    # ------------------------------------------------------------------
    # list_jobs（輔助，列出現有 Studio）
    # ------------------------------------------------------------------

    def list_jobs(self) -> list[dict]:
        """列出 Teamspace 中所有 Studio 狀態"""
        if not self.is_available():
            return []

        try:
            from lightning_sdk import Teamspace

            ts = Teamspace(name=LIGHTNING_TEAMSPACE, user=LIGHTNING_USERNAME)
            studios = ts.studios
            return [
                {
                    "studio_name": s.name if hasattr(s, "name") else str(s),
                    "status": str(getattr(s, "status", "unknown")),
                }
                for s in studios
            ]
        except Exception as e:
            logger.warning("list_jobs failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    def _resolve_machine(self, machine: str):
        """將字串機型解析為 Machine enum"""
        from lightning_sdk import Machine
        try:
            return getattr(Machine, machine.upper())
        except AttributeError:
            logger.warning("Unknown machine type '%s', fallback to T4", machine)
            return Machine.T4

    def _build_train_command(self, config: Optional[dict]) -> str:
        """從 config dict 組裝 yolo train 指令"""
        cfg = config or {}
        epochs = cfg.get("epochs", 50)
        imgsz = cfg.get("imgsz", 640)
        batch = cfg.get("batch", 16)
        model = cfg.get("model", "yolov8n.pt")
        data_yaml = cfg.get("data_yaml", "dataset/data.yaml")

        cmd = (
            f"yolo train "
            f"model={model} "
            f"data={data_yaml} "
            f"epochs={epochs} "
            f"imgsz={imgsz} "
            f"batch={batch} "
            f"device=0 "
            f"2>&1 | tee training.log"
        )
        return cmd

    def get_api_key_instructions(self) -> str:
        """回傳取得 API Key 的說明"""
        return (
            "Lightning AI 憑證取得步驟：\n"
            "1. 前往 https://lightning.ai\n"
            "2. 登入帳號（GitHub 登入）\n"
            "3. 右上角 → Settings → API Keys → Create API Key\n"
            "4. 複製 API Key 與 User ID，存入 Hurricane Vault：\n"
            "   hvault set hurricanecore/dev/LIGHTNING_USER_ID <user_id>\n"
            "   hvault set hurricanecore/dev/LIGHTNING_API_KEY <api_key>\n"
            "5. 啟動服務時使用：\n"
            "   hvault run --prefix 'hurricanecore/dev/' -- docker-compose up\n"
            "\n"
            "注意：Lightning AI 免費方案提供 22hr/月 GPU（T4），超量按使用計費。\n"
            "Teamspace：default-teamspace / User：cancleeric\n"
        )
