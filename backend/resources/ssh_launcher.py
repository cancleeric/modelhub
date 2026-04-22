"""
resources/ssh_launcher.py — SSH 訓練啟動器（Sprint 23 Task 23-1）

透過 SSH 連線到內網 GPU 主機執行訓練任務。
主機清單從 TRAINING_SSH_HOSTS env 取得（格式：user@host，逗號分隔）。
使用 SSH key 認證，不使用密碼。

SSH 命令：
  submit_job:    上傳 dataset + 執行訓練腳本（背景 nohup）
  get_job_status: 查詢 nohup pid 存活狀態
  download_output: 用 scp/rsync 下載 best.pt
"""

import base64
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("modelhub.resources.ssh_launcher")

SSH_TIMEOUT = int(os.environ.get("MODELHUB_SSH_TIMEOUT", "10"))
SSH_COMMON_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
]


class SSHLauncher:
    """
    SSH 遠端 GPU 訓練啟動器。

    資源優先序：Kaggle（主力）→ Lightning → SSH → 本機 MPS
    """

    def submit_job(
        self,
        host: str,
        req_no: str,
        dataset_path: str,
        config: Optional[dict] = None,
    ) -> dict:
        """
        在 SSH host 上啟動訓練任務。

        步驟：
        1. 建立遠端工作目錄 ~/modelhub-jobs/<req_no>/
        2. rsync dataset 到遠端
        3. 寫入 config.json
        4. nohup 執行訓練腳本（背景執行，PID 存入 job.pid）

        Args:
            host:         SSH 目標（格式：user@192.168.50.83）
            req_no:       需求單編號
            dataset_path: 本機 dataset 目錄路徑
            config:       訓練設定 dict

        Returns:
            {"success": True, "host": host, "job_dir": str} 或
            {"success": False, "reason": str}
        """
        cfg = config or {}
        remote_job_dir = f"~/modelhub-jobs/{req_no}"

        # 1. 建立遠端目錄
        mkdir_result = subprocess.run(
            ["ssh"] + SSH_COMMON_OPTS + [host, f"mkdir -p {remote_job_dir}"],
            capture_output=True, text=True, timeout=SSH_TIMEOUT,
        )
        if mkdir_result.returncode != 0:
            return {
                "success": False,
                "reason": f"mkdir failed on {host}: {mkdir_result.stderr.strip()[:200]}",
            }
        logger.info("submit_job: created remote dir %s:%s", host, remote_job_dir)

        # 2. rsync dataset
        dataset_local = Path(dataset_path)
        if dataset_local.exists():
            rsync_result = subprocess.run(
                [
                    "rsync", "-az", "--timeout=60",
                    str(dataset_local) + "/",
                    f"{host}:{remote_job_dir}/dataset/",
                ],
                capture_output=True, text=True, timeout=120,
            )
            if rsync_result.returncode != 0:
                logger.warning(
                    "submit_job: rsync warning for req=%s host=%s: %s",
                    req_no, host, rsync_result.stderr.strip()[:200],
                )
                # rsync 失敗不中斷，繼續嘗試（dataset 可能已在遠端）
        else:
            logger.warning("submit_job: dataset_path %s not found, skipping rsync", dataset_path)

        # 3. 寫入 config.json 到遠端
        config_json = json.dumps({
            "req_no": req_no,
            "epochs": cfg.get("epochs", 50),
            "imgsz": cfg.get("imgsz", 640),
            "batch": cfg.get("batch", 16),
            "arch": cfg.get("arch", "yolov8m"),
            "data_yaml": cfg.get("data_yaml", "dataset/data.yaml"),
        }, ensure_ascii=False)

        # P1-5: 用 base64 傳輸，避免 config_json 含單引號時 shell injection
        encoded = base64.b64encode(config_json.encode()).decode()
        config_cmd = f"echo {encoded} | base64 -d > {remote_job_dir}/config.json"
        subprocess.run(
            ["ssh"] + SSH_COMMON_OPTS + [host, config_cmd],
            capture_output=True, text=True, timeout=SSH_TIMEOUT,
        )

        # 4. 組裝訓練指令並以 nohup 背景執行
        train_cmd = self._build_train_command(remote_job_dir, cfg)
        nohup_cmd = (
            f"cd {remote_job_dir} && "
            f"nohup bash -c '{train_cmd}' "
            f"> {remote_job_dir}/training.log 2>&1 & "
            f"echo $! > {remote_job_dir}/job.pid && "
            f"echo started:$!"
        )
        run_result = subprocess.run(
            ["ssh"] + SSH_COMMON_OPTS + [host, nohup_cmd],
            capture_output=True, text=True, timeout=SSH_TIMEOUT,
        )
        if run_result.returncode != 0:
            return {
                "success": False,
                "reason": f"nohup failed on {host}: {run_result.stderr.strip()[:200]}",
            }

        output = run_result.stdout.strip()
        pid = None
        if output.startswith("started:"):
            try:
                pid = int(output.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

        logger.info(
            "submit_job: req=%s started on %s pid=%s job_dir=%s",
            req_no, host, pid, remote_job_dir,
        )
        return {
            "success": True,
            "host": host,
            "job_dir": remote_job_dir,
            "pid": pid,
        }

    def get_job_status(self, host: str, req_no: str) -> str:
        """
        查詢 SSH job 狀態（同步）。

        讀取遠端 job.pid，確認 process 是否存活。
        若有 done 標記檔（job.done）→ "complete"。
        若有 failed 標記檔（job.failed）→ "error"。

        Returns:
            "running" | "complete" | "error" | "unknown"
        """
        remote_job_dir = f"~/modelhub-jobs/{req_no}"

        check_cmd = (
            f"if [ -f {remote_job_dir}/job.done ]; then echo complete; "
            f"elif [ -f {remote_job_dir}/job.failed ]; then echo error; "
            f"elif [ -f {remote_job_dir}/job.pid ]; then "
            f"  pid=$(cat {remote_job_dir}/job.pid); "
            f"  if kill -0 $pid 2>/dev/null; then echo running; else echo error; fi; "
            f"else echo unknown; fi"
        )

        try:
            result = subprocess.run(
                ["ssh"] + SSH_COMMON_OPTS + [host, check_cmd],
                capture_output=True, text=True, timeout=SSH_TIMEOUT,
            )
            status = result.stdout.strip().lower()
            if status not in ("running", "complete", "error", "unknown"):
                status = "unknown"
            logger.debug("get_job_status: host=%s req=%s status=%s", host, req_no, status)
            return status
        except subprocess.TimeoutExpired:
            logger.warning("get_job_status: SSH timeout for host=%s req=%s", host, req_no)
            return "unknown"
        except Exception as e:
            logger.warning("get_job_status: exception for host=%s req=%s: %s", host, req_no, e)
            return "unknown"

    def download_output(self, host: str, req_no: str, local_dir: str) -> bool:
        """
        從 SSH host 下載訓練產出（best.pt + training.log）到 local_dir。

        Args:
            host:      SSH 目標（user@host）
            req_no:    需求單編號
            local_dir: 本機目標資料夾

        Returns:
            True 若成功下載至少一個檔案，False 若失敗
        """
        remote_job_dir = f"~/modelhub-jobs/{req_no}"
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # 嘗試找 best.pt（可能在 runs/train/exp/weights/ 下）
        find_cmd = f"find {remote_job_dir} -name 'best.pt' 2>/dev/null | head -1"
        try:
            find_result = subprocess.run(
                ["ssh"] + SSH_COMMON_OPTS + [host, find_cmd],
                capture_output=True, text=True, timeout=SSH_TIMEOUT,
            )
            remote_best_pt = find_result.stdout.strip()
        except Exception as e:
            logger.warning("download_output: find best.pt failed host=%s req=%s: %s", host, req_no, e)
            remote_best_pt = ""

        success = False

        # 下載 best.pt
        if remote_best_pt:
            local_best = str(local_path / "best.pt")
            try:
                scp_result = subprocess.run(
                    ["scp"] + SSH_COMMON_OPTS + [f"{host}:{remote_best_pt}", local_best],
                    capture_output=True, text=True, timeout=300,
                )
                if scp_result.returncode == 0:
                    logger.info("download_output: best.pt downloaded to %s", local_best)
                    success = True
                else:
                    logger.warning(
                        "download_output: scp best.pt failed: %s",
                        scp_result.stderr.strip()[:200],
                    )
            except Exception as e:
                logger.warning("download_output: scp best.pt exception: %s", e)

        # 下載 training.log
        remote_log = f"{remote_job_dir}/training.log"
        local_log = str(local_path / "training.log")
        try:
            scp_log = subprocess.run(
                ["scp"] + SSH_COMMON_OPTS + [f"{host}:{remote_log}", local_log],
                capture_output=True, text=True, timeout=60,
            )
            if scp_log.returncode == 0:
                logger.info("download_output: training.log downloaded to %s", local_log)
                success = True
        except Exception as e:
            logger.debug("download_output: training.log download failed: %s", e)

        return success

    def _build_train_command(self, job_dir: str, config: dict) -> str:
        """組裝遠端訓練指令"""
        epochs = config.get("epochs", 50)
        imgsz = config.get("imgsz", 640)
        batch = config.get("batch", 16)
        arch = config.get("arch", "yolov8m")
        data_yaml = config.get("data_yaml", f"{job_dir}/dataset/data.yaml")

        cmd = (
            f"yolo train "
            f"model={arch}.pt "
            f"data={data_yaml} "
            f"epochs={epochs} "
            f"imgsz={imgsz} "
            f"batch={batch} "
            f"device=0 "
            f"2>&1 | tee {job_dir}/training.log && "
            f"touch {job_dir}/job.done || touch {job_dir}/job.failed"
        )
        return cmd
