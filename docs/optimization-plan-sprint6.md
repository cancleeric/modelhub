# ModelHub 優化計劃 — Sprint 6

> 作者：Anderson（CEO 代 CPO 職）
> 日期：2026-04-16
> 狀態：草案 → CEO 已核可 → 派 CTO 開發
> 上一波：Sprint 2-5（v0.2.0 → v0.3.0）已完工，見 `sprint2-5-release.md`

---

## 為什麼要 Sprint 6

Sprint 2-5 留了三個「需 CEO 決策」項目：
1. Kaggle credential 還沒掛入 API container（poller 每分鐘空跑）
2. commtool/mailtool 不在 container 內（通知/週報 no-op）
3. actions router 只認 LIDS token（機器呼叫被擋）

再加上真實使用後會冒出的幾個痛點：
4. 訓練一失敗就停，業務方要手動 retrain
5. 提交時 Kaggle dataset URL 有沒有打錯沒人檢查，審核後才發現
6. 驗收工單多時一個一個點太慢
7. 沒有成本上限警示，GPU 時數可能意外爆量

Sprint 6 把這 7 件事一次做完。規模仍可控（估 2 天），不做架構大變動。

---

## 總覽

| # | 項目 | 類別 | 工時 |
|---|------|------|------|
| 6.1 | Kaggle credential 掛入 container | Plumbing | 0.5h |
| 6.2 | actions router 支援 API Key（雙軌 auth） | Plumbing | 0.5h |
| 6.3 | seed_data.py 相容新欄位 | Plumbing | 0.5h |
| 6.4 | 訓練失敗自動重試（可設上限） | Resilience | 3h |
| 6.5 | 資料集健康檢查（submit 前驗 URL + class 數） | Data Quality | 2h |
| 6.6 | 驗收儀表板（批次驗收 pending_acceptance） | UX | 3h |
| 6.7 | 成本預算警示（單張工單 max_budget_usd） | Cost | 2h |

---

## 6.1 Kaggle credential

### 問題
Poller 每 60 秒跑一次，但 `KAGGLE_USERNAME` / `KAGGLE_KEY` 未設，每次都 skip。

### 方案
docker-compose.yml 新增 volume 把 host 的 `~/.kaggle/kaggle.json` bind 進 container `/root/.kaggle/kaggle.json`。API key 走 file 而非 env（更安全，也符合 kaggle SDK 預設）。若 host 沒有該檔，volume 會變成 directory（導致報錯）→ 加 `MODELHUB_KAGGLE_ENABLED=true` env 當開關，預設不掛。

### 驗收
設好 `kaggle.json` 且 enable=true → poller 真正呼到 Kaggle API；未設時仍 skip 不 crash。

---

## 6.2 actions router 雙軌 auth

### 問題
reject/resubmit/attach-kernel 都是 `CurrentUser`（LIDS only），E2E 自動化跟跨 service 呼叫都會被擋。

### 方案
統一換成 `CurrentUserOrApiKey`，機器走 `X-Api-Key`、人走 Bearer token。

### 驗收
`curl -H 'X-Api-Key: ...' ... /reject` 能成功。actor 欄位 fallback 到 `"service_account"`。

---

## 6.3 seed_data.py

### 問題
舊 seed 不動新欄位（nullable OK），但若要種測試資料跨 reject/resubmit 情境，得加 history。

### 方案
補一條 demo submission 走過 submit → reject → resubmit → approve → attach-kernel，方便 PM 驗收時點進去看 UI。

### 驗收
`docker exec ... python seed_data.py` 後 MH-2026-XXX 的 /history tab 有 4+ 筆紀錄。

---

## 6.4 訓練失敗自動重試

### 問題
Poller 遇到 kernel error → status=failed，業務方要手動按「重新訓練」。常見失敗（OOM、timeout、資料下載失敗）自動重試就好。

### 方案
Submission 新增：
- `max_retries`（預設 2）
- `retry_count`（已重試次數）

Poller 偵測 `complete/error`：
- error 且 retry_count < max_retries → 自動 push kernel 重跑、`retry_count +=1`、不變 status
- error 且 retry_count >= max_retries → status=failed + 通知

Kernel 重跑用 `kaggle kernels push` CLI（需已下載過 kernel 本地副本）。

### 驗收
- max_retries=2 的單，error 兩次才 failed（第一次觸發重跑、第二次 failed）
- history 有 `training_retry` 紀錄
- 通知只在最終 failed 時發，中間 retry 不吵

---

## 6.5 資料集健康檢查

### 問題
業務方 copy-paste Kaggle URL 偶有打錯，CTO 要進系統點 link 才發現 404。

### 方案
`POST /api/submissions/` 時：
- 若 `kaggle_dataset_url` 非空 → httpx HEAD 確認 2xx
- 若 `class_list` 有值、`class_count` 有值 → 比對是否一致

不通過時**不擋**建單（可能是私有 dataset），但在 response 加 `warnings: [...]` 欄位，前端顯示黃色提示 banner。

### 驗收
- URL 404 → warning 列出「Kaggle URL 無法訪問」
- class_list=7 類但 class_count=5 → warning 列出「類別數不一致」
- 前端提交頁回應含 warnings 時顯示 banner

---

## 6.6 驗收儀表板（AcceptanceQueue）

### 問題
CEO/業務方要一個個點 submission → registry → accept。工單多時效率差。

### 方案
新頁 `/acceptance`：
- 列出所有 `status=pending_acceptance` 的 ModelVersion
- 表格列：req_no / 名稱 / mAP50_actual vs map50_target / 訓練時間 / 成本 / `通過 / 失敗` 兩顆按鈕
- 通過 → 呼叫既有 `/api/registry/{id}/accept`
- 批次通過：勾多個 + 一鍵全部 accept

### 驗收
- 有 3+ 個 pending_acceptance 時，頁面一次顯示全部
- 批次按鈕能 parallel 打 3 個 accept API 並成功

---

## 6.7 成本預算警示

### 問題
Kaggle 雖免費，但 P100 時數不控會卡別人。老闆想知道「這張單到底吃了多少」。

### 方案
Submission 加 `max_budget_usd`（預設 5.0）。

前端 Detail/Kaggle tab 顯示 progress bar：
- 綠：< 70% budget
- 黃：70-100%
- 紅：> 100%（並在 poller 偵測到超過時發 CTO 通知）

Poller 超預算 → `notify_event("budget_exceeded", ...)`。

### 驗收
- budget=$5、estimated=$3 → 綠條 60%
- estimated>$5 → 紅條 + CMC 通知（fallback warning OK）

---

## 風險

1. **6.4 自動重試的 kernel push**：需要 kernel 本地 metadata（kaggle 要求 `kernel-metadata.json`）。若沒留本地副本只能失敗 → 退而求其次：只重新執行同 kernel（`kaggle kernels pull` → `kaggle kernels push`），CTO 確認這個流程能跑。
2. **6.5 httpx HEAD 被 Kaggle 擋 403**：Kaggle 登入頁要 cookie。若 HEAD 不通改 GET /datasets/xxx 第一頁 HTML 看 title，再判斷。
3. **6.6 批次 accept**：map50_actual 是空的狀況（parser 沒抓到）→ 前端 disabled 通過按鈕，顯示「需手動填入」。

---

## 不做（推回 Sprint 7+）

- API key table + rotation（目前單 key 夠用）
- LLM 輔助審核（先用 rule-based warnings 取代）
- 部署自動銜接 HurricaneEdge（跨公司接口，需先談）
- 週報圖表化（email 純文字夠用）

---

## 結論

Sprint 6 工時約 2 天，全部獨立、互不阻塞。派 CTO 一次做完，交 CEO 統一審查。
