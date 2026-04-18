# Changelog — ModelHub

## [0.6.0] — 2026-04-18 (Sprint 20-24)

### Sprint 20 — 持久化訓練隊列（P0）
- **Task 20-1**：`models.py` 新增 `TrainingQueue` 資料表（id/req_no/priority/status/enqueued_at/dispatched_at/target_resource/retry_count/error_reason）
- **Task 20-1**：新增 `alembic/versions/0002_training_queue.py` migration（PostgreSQL）
- **Task 20-2**：新建 `backend/queue_manager.py`（enqueue/peek_next/mark_dispatching/mark_running/mark_done/mark_failed/count_running/get_queue_position）
- **Task 20-3**：`routers/actions.py` approve 段改為呼叫 `QueueManager.enqueue()`，移除直接 `background_tasks.add_task` 派發
- **Task 20-4**：新建 `pollers/queue_dispatcher.py`（每 30 秒，MAX_CONCURRENT=2），加入 main.py lifespan
- **Task 20-5**：新增 `GET /api/queue/status` endpoint（routers/queue.py），回傳 waiting/running 清單

### Sprint 21 — 定時健康檢查（P0）
- **Task 21-1**：新建 `pollers/health_checker.py`（每 10 分鐘，6 種警告類型，1 小時節流），加入 main.py lifespan
- **Task 21-2**：`routers/health.py` `/api/health/system-status` 擴充 queue_status/poller_health/success_rate_24h
- **Task 21-3**：`health_checker.py` 新增 `send_daily_report()`，APScheduler cron 08:00 Asia/Taipei 每日發送

### Sprint 22 — Lightning 完整實作（P0）
- **Task 22-1**：`_normalize_status()` 完整確認，新建 `tests/test_lightning_status_mapping.py`（19 測試全過）
- **Task 22-2**：`resources/prober.py` 新增 `LightningQuotaTracker`（get_used_hours_this_month/get_remaining_hours/is_quota_available），整合到 `probe_lightning()`（配額耗盡時回傳 available: False）
- **Task 22-3**：新建 `tests/test_lightning_e2e.py`（19 測試全過，mock SDK）
- **Task 22-4**：`lightning_poller.py` poll_once() 加入 `_check_lightning_quota_warning()`

### Sprint 23 — SSH 訓練啟動器（P1）
- **Task 23-1**：新建 `resources/ssh_launcher.py`（submit_job/get_job_status/download_output，nohup + job.pid + job.done/job.failed 機制）
- **Task 23-2**：`routers/actions.py` SSH 分支實際呼叫 `SSHLauncher.submit_job()`，失敗時 fallback local
- **Task 23-3**：新建 `pollers/ssh_poller.py`（120s，complete/error/overtime 處理，queue 狀態同步），加入 main.py lifespan

### Sprint 24 — 前端整合（P1）
- **Task 24-1**：`SubmissionListPage.tsx` status=approved 顯示「排隊中 #N」/「派發中」，呼叫 `GET /api/queue/status`（30s refresh）
- **Task 24-2**：新建 `components/ResourceHealthWidget.tsx`（60s refresh，Kaggle/Lightning 配額進度條、任務數、poller 健康、24h 成功率），整合到 StatsPage
- **Task 24-3**：新建 `pages/QueuePage.tsx`（waiting/running 清單，reviewer 可調整優先序），加入 App.tsx + Layout.tsx 導航

### Sprint 25 QA
- `pytest tests/ -v` — 38/38 通過
- 版本號從 0.5.0 bump 到 0.6.0

---

## [0.5.0] — 2026-04-17 (Sprint 8)

### 新增功能
- **8.1 推論 server 持久化**：macOS LaunchAgents plist，KeepAlive=true，kill 後自動重啟
  - `launchd/tw.hurricanecore.modelhub.inference.plist`
  - `scripts/inference-service.sh` 管理腳本
  - inference_server.py port 改從 `MODELHUB_INFERENCE_PORT` env 讀取
- **8.2 訓練完成自動回寫 DB**：訓練腳本尾部自動呼叫 API 更新 submission status
  - `training/modelhub_report.py` 共用 hook
  - 後端新增 `PATCH /api/submissions/{req_no}/training-result` endpoint
  - mh-2026-006、mh-2026-008 train.py 整合 report_result
- **8.3 推論測試 UI 頁面**：`/predict` 新頁面
  - 下拉選 active 模型、拖放上傳圖片、信心度 bar chart
  - Layout NAV 加入「推論測試」入口
- **8.4 mh-2026-006 Resume 重訓**：從 best.pt checkpoint 繼續 20 epochs
  - `training/mh-2026-006/resume_train.py`
  - 結果寫入 `result_v2.json`，自動呼叫 modelhub_report

### 技術改善
- 加入 `.gitignore`（排除 .pt/.pth 大型模型檔、訓練輸出目錄、logs）

---

## [0.4.0] — Sprint 7

### 功能
- API Key 管理頁面（DB-backed，Sprint 7.1）
- 版本號單一來源 VERSION 檔（Sprint 7.5）
- bump-version.sh 工具腳本

---

## [0.3.1] — Sprint 6

### 功能
- 自動重試機制（max_retries / retry_count）
- 預算控制（max_budget_usd）

---

## [0.3.0] — Sprint 2-3

### 功能
- 結構化退件（rejection_reasons）
- Kaggle Kernel 整合（kaggle_status）
- 訓練時程追蹤（training_started_at / training_completed_at）
