# ModelHub 優化計劃 — Sprint 7

> 作者：Anderson（CEO 代 CPO 職）
> 日期：2026-04-16（Sprint 6 當日完工後續作）
> 狀態：草案 → CEO 核可 → 派 CTO 開發
> 前兩波：Sprint 2-5（v0.3.0）、Sprint 6（v0.3.1）

---

## 為什麼要 Sprint 7

Sprint 6 補完了 plumbing，現在系統**能跑**；Sprint 7 要讓它**像個產品**。

目前真實問題：
1. API Key 寫死在 env（一把走天下，無法多方授權、無法撤銷）→ 安全性
2. 提交時 PM 常漏填欄位，warnings 只能抓客觀錯誤（404/數字不符）→ 沒有智能建議
3. 模型清冊沒有「當前使用中」標記，PM 要翻資料才知道哪顆模型現役 → UX
4. 沒有統計頁，老闆要看活躍度得自己 SQL → 管理視角缺失

Sprint 7 解這四件事。工時估 2 天，不碰跨公司邊界，不動既有資料結構。

---

## 總覽

| # | 項目 | 類別 | 工時 |
|---|------|------|------|
| 7.1 | API Key table + rotation | Security | 3h |
| 7.2 | LLM 輔助審核（走 Anemone → brain） | Product | 3h |
| 7.3 | Registry UX：filter + 當前版本 badge | UX | 1.5h |
| 7.4 | 統計 Dashboard `/stats` 頁 | Management | 2.5h |

---

## 7.1 API Key table + rotation

### 問題
`MODELHUB_API_KEY=modelhub-dev-key-2026` 寫死 env，沒人看得到誰在用、無法撤銷某家、泄露要全體換。

### 方案

新表：
```python
class ApiKey(Base):
    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, index=True, nullable=False)  # 實際 token
    name = Column(String, nullable=False)         # 如 "AICAD pipeline" / "天機 api"
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    disabled = Column(Boolean, default=False)
```

`auth.py` 改：
- 先從 DB 找 key match + disabled=false → 更新 `last_used_at`
- 找不到 fallback 檢查 env bootstrap key（保留用於首次管理）

新 router `routers/api_keys.py`：
- `GET  /api/admin/api-keys` — 列出（bearer token only，需登入）
- `POST /api/admin/api-keys {name}` — 建新 key（server gen 32 byte urlsafe）
- `POST /api/admin/api-keys/{id}/disable` — 停用（不 delete 保留稽核）

### 驗收
- bootstrap key 仍可用
- 新建 key 可驗證；disable 後 401
- `last_used_at` 正確更新

---

## 7.2 LLM 輔助審核

### 問題
目前 warnings 只是 rule-based（URL HEAD / class 數對不對 / mAP 範圍）。PM 還是漏：缺業務問題描述、purpose 太模糊、class_list 命名不合標、交付日不合理。

### 方案
Submit 時呼叫 LLM 做「pre-review」：
1. 組一個 prompt：「這是一張模型訓練需求單，請找出 3-5 個潛在風險或遺漏（如資料量太少、類別設計不合理、mAP 目標與資料量不匹配）」
2. 走集團規範：**HTTP 到 Anemone ingress**（`ANEMONE_API_URL` + `ANEMONE_API_KEY` env），不直連 brain，不直連 LLM 廠商
3. 回傳 suggestions 跟既有 warnings 一起塞到 create response
4. Anemone 不可達時 skip，不 fail submit

### 實作重點
- 新檔 `backend/advisors/llm_advisor.py`
- 呼叫 Anemone：`POST /v1/chat/completions` with `call_type: "modelhub/submission-review"`（走集團 call_type 分類）
- 使用 10 秒 timeout，失敗/超時都 skip
- 回傳格式：`{"suggestions": ["...", "..."]}`
- SubmissionCreate response 加 `suggestions: string[]` 欄位

### 驗收
- Anemone 可達時，submit 回傳含 1-5 條 LLM 建議
- Anemone 不可達時，submit 照樣成功，suggestions=[]
- 前端顯示「智能建議」區塊（跟 warnings 分開）

---

## 7.3 Registry UX

### 問題
模型清冊只能看 ID + 狀態，搜不到想要的。is_current 欄位後端有但前端無顯示。

### 方案
- Registry 頁加 filter：status (active/pending_acceptance/retired/testing)、product、req_no 搜尋
- 每列加「當前」badge（is_current=true 時綠色 pill）
- pending_acceptance 行點進去直接跳 `/registry/{id}/accept`

### 驗收
- 有 10+ 版本時，filter 能快速找到 AICAD 下的當前模型
- 批次驗收後回到 registry 能一眼看到新的 active

---

## 7.4 統計 Dashboard

### 問題
老闆問「這個月跑了幾個模型？哪個 product 最多？平均花多久？」沒答案。

### 方案
新頁 `/stats`，純前端從既有 API 組：
- `GET /api/submissions/` + `GET /api/registry/` 抓全部，client 端 aggregate
- 指標卡：
  - 本月新 submission / 完成訓練 / 通過驗收
  - 平均從提交到 accepted 的時數
  - 累計 GPU 時數 + 成本
  - 各 product 分布（bar）
  - 各 company 分布（bar）
- 後端不加新 endpoint（保持簡潔）

### 驗收
- 頁面 1 秒內渲染，資料從真 API 來
- 數字可對照 CEO 週報

---

## 不做（Sprint 8+ 儲備）

- accepted → Anemone 事件通知 HurricaneEdge：跨公司接口，要先談
- 週報 HTML email：純文字夠老闆用
- 審核 SLA 警示：目前工單量還不需要
- commtool 進 container：本機通知 fallback 可接受

---

## 風險

1. **7.1 bootstrap key fallback**：DB 空表 + 有人打 api → 必須 fallback 環境 key，否則連管理都進不去。CTO 要特別測：空 DB + env key → 200。
2. **7.2 Anemone 可達性**：dev 容器用 squid_dev_network，Anemone server 應該在同網段。若 dev env 沒起 Anemone，LLM advisor 全失敗 → 前端要優雅處理。
3. **7.4 純前端 aggregate**：若 submission 數量 > 1000 拉全清單會慢。目前還在個位數，先不管 pagination。

---

## 結論

Sprint 7 四項獨立，全部在本公司內部不跨邊界。派 CTO 一次完成，CEO 統一審查。
