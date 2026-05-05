# Changelog — ModelHub

## [0.7.2] — 2026-05-05 (Sprint 34)

### Model Lifecycle Updates
- **MH-2026-025 FP Ranker v2 Promoted**：Shadow 期滿（7 天）→ Promote，AUC=0.814；submission status=deployed，model_version status=active
- **MH-2026-026 Severity Classifier v4 Rejected**：accuracy=25.0% < 50% threshold，submission status=rejected，model_version notes 補記失敗原因
- **MH-2026-030 Probe Quality Scorer 資料不足 Rejected**：補建 model_version v1（pass_fail=fail），notes=data_analysis_only，需 50+ samples 才可訓練

### Aegis ML Pipeline
- **MH-2026-031 Prompt Injection Classifier**：解除 blocked，attach Kaggle kernel `boardgamegroup/aegis-prompt-injection-classifier`，status=training
- **MH-2026-032 Code Vulnerability Detector**：解除 blocked，attach Kaggle kernel `boardgamegroup/aegis-code-vuln-detector`，status=training
- **MH-2026-034 LLM Output Toxicity Classifier**：新增 submission 登記，status=pending（待建 Kaggle dataset），kernel stub 建立於 `kaggle-kernels/aegis-llm-output-toxicity/`

### Bug Fix & Enhancement
- **Kaggle stale job 偵測（S34-07）**：`kaggle_poller._detect_stale_jobs()` — 查 `kaggle_status=running` 且 `kaggle_status_updated_at < NOW()-72h` 的 submission，自動標記 `stale_timeout`，通知 CTO；排程每 6 小時執行
- **`poll_once` 每次 poll 後無條件更新 `kaggle_status_updated_at`**：不論狀態是否變化，確保 stale 偵測有準確時間戳記
- **新增環境變數 `MODELHUB_KAGGLE_STALE_TIMEOUT_HOURS`**（預設 72）

### Tests
- 新增 `TestDetectStaleJobs`（4 tests）：stale 常數驗證、`kaggle_status_updated_at` 無條件更新驗證、無 stale job 回傳 0、stale job 正確標記 stale_timeout
- 212 passed（含新增 4 tests）

---

## [0.7.1] — 2026-05-04 (Sprint 33)

### New Features
- **Kaggle 自動驗收（P0）**：`kaggle_poller._auto_accept()` — kernel complete 且 `pass_fail=pass` 時，自動打 `POST /api/registry/{mv_id}/accept`，並透過 CMC 通知 CTO
- **輕量 Kaggle 狀態 Endpoint**：`GET /api/submissions/{req_no}/kaggle-status`（`submissions.py`）— 無需 auth，只查 DB，回傳 `req_no/kaggle_status/kaggle_status_updated_at/kaggle_kernel_slug`
- **前端 Kaggle Tab Auto-refresh**：`SubmissionDetailPage.tsx` — `kaggle_status=running` 且在 kaggle tab 時，每 30 秒打輕量 endpoint 更新顯示；`complete` 或 `error` 時停止 polling 並 invalidate 主查詢

---

## [0.7.0] — 2026-05-04 (Sprint 32)

### Bug Fixes
- **MH-2026-029 CER normalization fix（P0）**：`kaggle_poller._parse_result_obj()` 新增防禦性 clamp，確保 `ocr_cer` 回傳值 <= 1.0；修復 Kaggle kernel 未正規化 CER（如 86.25）導致指標異常的問題
- **kaggle-kernels/ocr_v2_kernel/train_kaggle.py** `compute_cer()` 尾端加 `min(cer_value, 1.0)` clamp，從 kernel 端防止產生超界 CER

### Tests
- 新增 `TestCerNormalization`（4 tests）：覆蓋 86.25 未正規化 CER、正常範圍不影響、略超 1.0 情境、`test_cer` 欄位同樣被 clamp

## [0.6.3] — 2026-05-04 (Sprint 29-31)

### New Features
- **Sprint 29**：`registry.accept_version` endpoint 開放 API Key 認證（`CurrentUserOrApiKey`），CTO 機器驗收無需 LIDS Bearer token
- **Sprint 29**：`registry.create_version` / `registry.update_version` 同步改為 `CurrentUserOrApiKey`
- **Sprint 29**：`accept_version` 驗收後自動同步 `ModelVersion.status`（pass → `active`，fail → `rejected`）
- **Sprint 30**：`AcceptancePayload` 新增 `pass_fail` 選填欄位，CTO 人工判定優先於自動 map50_target 比較；`map50_actual` 改為選填

### Model Acceptance (2026-05-04)
- MH-2026-022 Aegis Severity Auto-classifier v1: accepted (accuracy=0.9662, f1=0.9673)
- MH-2026-027 PPE Detection v3 yolov8l v1: accepted (mAP50=0.8996)
- MH-2026-019 Site Object Detection v1: accepted (mAP50=0.7455); v2 (0.4322) rejected
- MH-2026-006 Text Detection v2: accepted CTO manual pass (mAP50=0.5993); v3 (0.4489) rejected
- MH-2026-008 Multiview Boundary v1: rejected (mAP50=0.6839 < 0.70 threshold)

### Tests
- 222 tests passed

---

## [0.6.2] — 2026-05-04 (Sprint 26-28)

### Bug Fixes
- **Sprint 26**：`GET /api/submissions/*` 和 `GET /api/registry/*` 改為接受 API Key（`CurrentUserOrApiKey`），讓機器對機器查詢無需 LIDS Bearer token
- **Sprint 26**：`pytest.ini` 新增 `asyncio_default_fixture_loop_scope=function`，消除 PytestDeprecationWarning
- **Sprint 26**：`lightning_poller.start_scheduler` 補充啟動時一次性 API Key 狀態說明
- **Sprint 27**：`kaggle_poller._on_kernel_error` 幂等保護修正：terminal status 清單補入 `training_failed`（原只有 `failed`），修復 retry 耗盡後 status 未正確設為 `training_failed` 的 bug
- **Sprint 27**：`lightning_poller._process_submission` 補充幂等保護
- **Sprint 27**：新增 `TestOnKernelErrorIdempotent` 測試類（3 個測試）
- **Sprint 28**：`kaggle_poller._append_training_failed_summary` 移除多餘 `actor` 參數，修復 `refresh-kaggle` endpoint 回 500 Internal Server Error 的問題
- **Sprint 28**：`attach-kernel` endpoint 加入 `training_failed` 狀態允許，失敗後可直接 attach 新 kernel 重訓（自動 reset retry_count=0）

### Tests
- 222 tests passed (前 219 + 3 新增)

---

## [0.6.1] — 2026-05-02 (feat/m23-phase4 初版)

### Sprint m23-phase4
- **前端四模組**：Comments/Attachments/Notifications/ExternalModels 頁面完成
- **Kaggle log fallback**：`##RESULT_JSON##` 標記機制，解決 result.json 無法下載問題
- **SSH 訓練追溯欄位**：補充 SSH 訓練相關欄位
- **Pydantic warnings**：消除 Pydantic v2 deprecation warnings
- **MH-029 開訓**：Engineering OCR TrOCR v2 開始 Kaggle 訓練
- 219 tests passed

---

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
