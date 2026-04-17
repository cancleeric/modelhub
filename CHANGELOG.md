# Changelog — ModelHub

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
