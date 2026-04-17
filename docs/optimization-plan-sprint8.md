---
title: ModelHub 優化計劃 — Sprint 8
description: 推論 server 持久化、訓練自動回寫 DB、推論測試 UI、006 微調重訓、版本號 bump v0.5.0
published: true
date: 2026-04-17
tags: [modelhub, sprint8, inference, automation, ux]
---

# ModelHub 優化計劃 — Sprint 8

> 作者：CPO（颶核科技）
> 日期：2026-04-17
> 狀態：草案 → CEO 核可 → 派 CTO 開發
> 前置版本：Sprint 7（v0.4.0）已完工

---

## 為什麼要 Sprint 8

Sprint 7 讓 ModelHub「像個產品」；Sprint 8 要讓它**在生產環境站穩腳跟**，並把當前 AICAD 訓練現況的兩個立即痛點解掉。

真實問題：
1. 推論 server（`:8951`）是手動背景跑，每次重開機要人工恢復，010/011 模型等於「隨時可能離線」
2. 每次訓練完成要手動 `docker exec` 改 submission status，工程師累積手動步驟多容易出錯
3. 只能 `curl` 測推論，老闆與 Rachel 要看效果需要工程師協助
4. mh-2026-006 文字偵測差 0.003 就可以 pass，放棄可惜
5. 以上四項完成後功能面已是 minor version 等級，需要 bump v0.5.0

痛點 7（模型無版本管理）與痛點 3（005/008 auto-label 改 SAM）、痛點 4（009 換 CRNN+CTC 架構）**不進本 Sprint**，原因見「不做」區塊。

---

## 總覽

| # | 項目 | 類別 | 工時 |
|---|------|------|------|
| 8.1 | 推論 server 持久化 + launchd | 穩定性 | 2h |
| 8.2 | 訓練完成自動回寫 DB | 自動化 | 2.5h |
| 8.3 | 推論測試 UI 頁面 | UX | 3h |
| 8.4 | mh-2026-006 微調重訓 | 訓練執行 | 1h（設定）+ 訓練時長 |
| 8.5 | 版本號 bump v0.5.0 | 版控 | 0.5h |

總估時：約 9h（不含 8.4 訓練等待時間）

---

## 8.1 推論 server 持久化 + launchd

### 目標
確保 inference_server.py 在 Mac 重開機後自動啟動，010/011 推論 API 不需人工介入即可持續服務。

### 背景
目前 inference_server.py 用 `nohup python inference_server.py &` 手動背景啟動，跑在 host `:8951`。一旦主機重開機或程序被 OOM kill，推論 API 全斷，ModelhHub 的 `/api/predict/{req_no}` proxy 就回 502。

### 實作規格

**launchd plist（macOS 原生 Service Manager）**

建立 `/Library/LaunchDaemons/tw.hurricanecore.modelhub.inference.plist`（系統層級，重開機自動起）：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>tw.hurricanecore.modelhub.inference</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/venv/bin/python</string>
        <string>/Users/yinghaowang/HurricaneCore/modelhub/inference_server.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/yinghaowang/HurricaneCore/modelhub/logs/inference_server.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/yinghaowang/HurricaneCore/modelhub/logs/inference_server_err.log</string>
    <key>WorkingDirectory</key>
    <string>/Users/yinghaowang/HurricaneCore/modelhub</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MODELHUB_INFERENCE_PORT</key>
        <string>8951</string>
    </dict>
</dict>
</plist>
```

**inference_server.py 修改要點**：
- 加上 `/health` endpoint，回傳 `{"status": "ok", "loaded_models": [...]}`
- 啟動時列印已載入的模型清單（從 modelhub DB 查 active 模型路徑）
- Port 改從 env `MODELHUB_INFERENCE_PORT`（預設 8951）讀取

**管理腳本** `scripts/inference-service.sh`：
```bash
# 安裝
sudo launchctl load /Library/LaunchDaemons/tw.hurricanecore.modelhub.inference.plist

# 啟停
sudo launchctl start tw.hurricanecore.modelhub.inference
sudo launchctl stop  tw.hurricanecore.modelhub.inference

# 查狀態
sudo launchctl list | grep modelhub
```

### 檔案變更

| 路徑 | 異動 |
|------|------|
| `modelhub/inference_server.py` | 加 `/health`、env port、啟動日誌 |
| `modelhub/scripts/inference-service.sh` | 新增管理腳本 |
| `modelhub/launchd/tw.hurricanecore.modelhub.inference.plist` | 新增（安裝時 sudo cp 到 /Library/LaunchDaemons/）|
| `modelhub/logs/` | 建立目錄（gitignore logs/*.log）|
| `modelhub/docs/runbooks/inference-server-ops.md` | 新增操作手冊 |

### 驗收標準
1. `sudo reboot`（或 `sudo launchctl start`）後，`curl http://localhost:8951/health` 在 30 秒內回 200
2. `POST /api/predict/mh-2026-010` 與 `POST /api/predict/mh-2026-011` 回傳推論結果
3. logs 有正確輸出至 `modelhub/logs/inference_server.log`

---

## 8.2 訓練完成自動回寫 DB

### 目標
訓練腳本（train.py）完成後自動呼叫 ModelhHub API 更新 submission status，消滅手動 `docker exec` 步驟。

### 背景
目前訓練結束後需人工進容器執行 SQL 或 API call 才能把 status 從 `training` 改成 `trained`（pass）或 `training_failed`（fail）。每張工單都要手動，不漏掉的前提是工程師記得。

### 實作規格

**通用回寫 hook**：新增 `training/modelhub_report.py`

```python
import os, requests

def report_result(
    req_no: str,
    passed: bool,
    metrics: dict,          # e.g. {"mAP50": 0.977, "epochs": 50}
    model_path: str | None = None,
    notes: str = ""
):
    """訓練腳本尾部呼叫此函式，自動回寫 ModelhHub DB"""
    base = os.environ.get("MODELHUB_API_URL", "http://localhost:8000")
    key  = os.environ.get("MODELHUB_API_KEY", "")
    payload = {
        "status": "trained" if passed else "training_failed",
        "metrics": metrics,
        "model_path": model_path,
        "notes": notes,
    }
    resp = requests.patch(
        f"{base}/api/submissions/{req_no}/training-result",
        json=payload,
        headers={"X-API-Key": key},
        timeout=10,
    )
    resp.raise_for_status()
    print(f"[modelhub_report] {req_no} → {payload['status']}")
```

**後端新增 endpoint**：`PATCH /api/submissions/{req_no}/training-result`
- 驗 API Key
- 允許 body：`status`（trained / training_failed）、`metrics`（JSON）、`model_path`、`notes`
- 同時更新 `submission.updated_at`
- 若 status=trained 且 model_path 有值，自動在 registry 建 pending_acceptance 記錄

**各 train.py 修改**：
- 在訓練迴圈尾部 import 並呼叫 `modelhub_report.report_result(...)`
- 優先改：mh-2026-007、mh-2026-008（進行中 / 即將開始的工單）
- 已完成工單（005/006/009/010/011）只需補 note 不需改腳本

**env 設定**：
訓練容器（或 host 訓練環境）需設定：
```
MODELHUB_API_URL=http://localhost:8000
MODELHUB_API_KEY=<從 Sprint 7 API Key 管理頁產生>
```

### 檔案變更

| 路徑 | 異動 |
|------|------|
| `training/modelhub_report.py` | 新增通用 hook |
| `backend/routers/submissions.py` | 新增 `PATCH /{req_no}/training-result` |
| `backend/schemas.py` | 新增 `TrainingResultUpdate` schema |
| `training/mh-2026-007/train.py` | 尾部加 report_result 呼叫 |
| `training/mh-2026-008/train.py` | 尾部加 report_result 呼叫 |

### 驗收標準
1. 執行 mh-2026-007 或 mh-2026-008 的 `train.py`，訓練結束後無需手動操作，DB 中 status 自動更新
2. ModelhHub 前端 submission 詳情頁顯示正確 status 與 metrics
3. 若 API 不可達（CI/離線環境），train.py 回寫失敗只 print warning，不中斷訓練主流程

---

## 8.3 推論測試 UI 頁面

### 目標
在 ModelhHub 前端新增「推論測試」頁面，讓老闆與 Rachel 不需 curl 即可上傳圖片、呼叫推論 API、看結果。

### 背景
目前推論只能靠 `curl -F image=@xxx.jpg http://localhost:8000/api/predict/mh-2026-010`。老闆或產品驗收時需要工程師代操，浪費溝通成本。有了 UI 後，非技術人員也能獨立驗收模型效果。

### 實作規格

**新增路由**：`/inference-test`（前端 SPA 新頁）

**頁面元件**：

```
┌─────────────────────────────────────────────┐
│ 推論測試                                     │
│                                             │
│ 選擇模型   [下拉選單：僅顯示 status=active]   │
│            ↳ 顯示：req_no + product + 說明   │
│                                             │
│ 上傳圖片   [拖放區 / 點選上傳]               │
│            ↳ 預覽縮圖                        │
│                                             │
│            [送出推論]                        │
│                                             │
│ 結果                                         │
│ ┌───────────────────────────────────────┐   │
│ │ 預測類別：PID_VALVE                   │   │
│ │ 信心度：0.977                         │   │
│ │ 推論時間：83ms                        │   │
│ │ [若 detection：圖片上繪製 bounding box]│   │
│ └───────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**技術選型**：
- 與現有前端技術棧一致（React / Vue，依現有 stack 而定）
- 圖片 bounding box 繪製用 Canvas API
- 檔案上傳用 FormData，直接 POST 到 `POST /api/predict/{req_no}`

**後端調整**（推論 API response 標準化）：
目前 `/api/predict/{req_no}` 直接 proxy 到 `:8951`，回傳格式視模型而定（classification vs detection 不同）。需在 proxy 層做輕量 normalize：

```json
{
  "req_no": "mh-2026-010",
  "model_type": "classification",   // "classification" | "detection" | "segmentation"
  "prediction": {
    "class": "PID_VALVE",
    "confidence": 0.977
  },
  "inference_ms": 83,
  "raw": { ... }   // inference_server 原始回傳，備查
}
```

Detection 型另加：
```json
"boxes": [
  {"class": "instrument", "confidence": 0.91, "bbox": [x1, y1, x2, y2]},
  ...
]
```

**導覽列**：在現有 sidebar 加入「推論測試」入口，Icon 用燒杯或閃電符號。

### 檔案變更

| 路徑 | 異動 |
|------|------|
| `frontend/src/pages/InferenceTest.vue（或 .tsx）` | 新增頁面元件 |
| `frontend/src/router/index.js` | 加 `/inference-test` 路由 |
| `frontend/src/components/Sidebar.vue` | 加導覽入口 |
| `backend/routers/predict.py` | response normalize 層 |
| `modelhub/inference_server.py` | response 補 `model_type` 欄位 |

### 驗收標準
1. 訪問 `/inference-test`，下拉選單列出所有 `status=active` 的模型（至少顯示 010、011）
2. 上傳一張測試圖片，點送出後顯示預測類別與信心度
3. Detection 型模型回傳時，圖片上正確繪製 bounding box
4. 非技術人員（老闆自測）能獨立完成一次推論，不需工程師說明

---

## 8.4 mh-2026-006 微調重訓

### 目標
mh-2026-006 文字偵測模型 mAP=0.5974，距 baseline 0.60 差 0.003，透過 resume + 延伸訓練 20 epochs 嘗試突破。

### 背景
006 訓練已跑完，last checkpoint 應存於 `training/mh-2026-006/yolo_run/weights/last.pt`。差距極小（0.5%），resume 繼續優化的成本遠低於重新標注。

### 實作規格

**Resume 指令**：
```bash
cd training/mh-2026-006
python train.py \
  --resume yolo_run/weights/last.pt \
  --epochs 20 \
  --patience 10
```

**train.py 調整**：
- 加 `--resume` 參數支援（YOLOv8 原生支援，確認 train.py 已 pass through 或直接呼叫 `model.train(resume=True)`）
- 完成後呼叫 `modelhub_report.report_result`（8.2 完工後立即整合）

**判定標準**：
- 20 epochs 後 mAP50 >= 0.60 → status=trained（pass），進驗收
- 若仍未達 → status=training_failed，blocked_reason 更新為「resume 20ep 仍未達，建議：(1) 增加標注資料 (2) 換 YOLOv8m 以上規格」

**注意**：此項是訓練執行，非開發工作。設定時間 1h，但訓練本身可能跑 2-4h（視 MPS 速度），安排在 8.1-8.3 開發期間並行進行。

### 檔案變更

| 路徑 | 異動 |
|------|------|
| `training/mh-2026-006/train.py` | 加 `--resume` 參數（如尚未支援）|
| `training/mh-2026-006/resume_run/` | 新 run 目錄，不覆蓋原始 yolo_run |

### 驗收標準
1. Resume 訓練正常啟動（從 last.pt 繼續，不是重頭跑）
2. 訓練結束後 modelhub_report 自動回寫 DB（依賴 8.2 完成）
3. 若 mAP >= 0.60：ModelhHub 顯示 mh-2026-006 status=trained，進入驗收流程
4. 若 mAP < 0.60：blocked_reason 更新，工單清楚說明後續選項

---

## 8.5 版本號 bump v0.5.0

### 目標
Sprint 8 功能面達到 minor version 等級（新 UI 頁面、新 API endpoint、持久化基礎設施），正式標記 v0.5.0。

### 實作規格
- `backend/version.py`（或等效位置）將 `__version__` 改為 `"0.5.0"`
- `CHANGELOG.md` 新增 v0.5.0 區塊，列出 8.1-8.4 完成項目
- Git tag：`v0.5.0`
- `GET /api/version` 回傳新版本號（若有此 endpoint）

### 檔案變更

| 路徑 | 異動 |
|------|------|
| `backend/version.py`（或 `__init__.py`）| `0.4.0` → `0.5.0` |
| `CHANGELOG.md` | 新增 v0.5.0 區塊 |

### 驗收標準
1. `GET /api/version` 回傳 `{"version": "0.5.0"}`
2. Git tag `v0.5.0` 存在
3. CHANGELOG 有完整的 Sprint 8 異動記錄

---

## 不做（延後至 Sprint 9+）

| 項目 | 理由 |
|------|------|
| 模型版本管理（v1/v2 概念）| 需要設計 model artifact storage 策略，影響 registry schema，工時大，Sprint 9 獨立規劃 |
| 005/008 auto-label 改 SAM | SAM 部署複雜度高，且 008 還在訓練中，等結果出來再決定是否啟動 |
| 009 換 CRNN+CTC 架構 | 需要全新訓練框架（非 YOLO），需要 CTO 評估工時，另開 Sprint |
| accepted → Anemone 跨公司通知 | 跨公司介面需先與 HurricaneEdge 協商 API 合約 |
| 週報 HTML email | 純文字夠用，不值得花 Sprint 時間 |

---

## 風險與相依性

| 風險 | 影響 | 對策 |
|------|------|------|
| 8.1 launchd plist 路徑寫死 venv | plist 中 python 路徑若與實際 venv 不符會靜默失敗 | 安裝腳本自動偵測 `which python`，寫入 plist |
| 8.2 endpoint 上線前 train.py 執行完 | mh-2026-007/008 訓練中，若 8.2 開發期間訓練結束，仍需一次手動回寫 | 手動回寫仍可行，8.2 重點是保護未來工單 |
| 8.3 inference response 格式未標準化 | 現有 010/011 的 raw response 格式可能不同，normalize 層需個別適配 | CTO 先查 inference_server.py 實際回傳，若差異大則 normalize 層改用 adapter pattern |
| 8.4 resume 後 mAP 仍未達 | 006 工單繼續失敗 | 此為可接受結果，blocked_reason 更新後 CEO 決定後續策略 |

---

## 執行順序建議

```
Day 1 上午：8.1（launchd）+ 同時啟動 8.4 resume 訓練（背景跑）
Day 1 下午：8.2（回寫 hook + endpoint）
Day 2 上午：8.3（推論測試 UI）
Day 2 下午：8.5（版號 + changelog + tag）+ 8.4 確認訓練結果
```

---

## 結論

Sprint 8 五項圍繞「讓已有的模型好用、穩定、可觀察」，不做新架構、不跨公司邊界。完成後 ModelhHub 從「工程師用的訓練管理後台」升級為「老闆與產品驗收人員也能直接使用的推論平台」，對應版本號 v0.5.0。

派 CTO 執行，CEO 統一審查驗收。
