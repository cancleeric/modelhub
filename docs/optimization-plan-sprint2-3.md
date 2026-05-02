# ModelHub 優化計劃 — Sprint 2 / 3 / 4+

> 作者：Anderson（CEO 代 CPO 職）
> 日期：2026-04-16
> 狀態：草案，待 CEO 審核後派發

---

## 總覽

| Sprint | 範圍 | 複雜度 | 預估工時 | 負責 |
|--------|------|--------|----------|------|
| Sprint 2 | 結構化退件 + 補件追蹤 | M | 2 天 | CTO |
| Sprint 3 | Kaggle kernel 狀態整合 | L | 3 天 | CTO |
| Sprint 4 | 訓練結果自動回填 + 成本追蹤 | M | 2 天 | CTO |
| Sprint 5 | 通知 + 報表 | S | 1 天 | CTO |

**總工時：8 工作天**（一個副手連續開發約 2 週）

---

## Sprint 2 — 結構化退件 + 補件追蹤

### 業務目標
目前退件只有一段文字（`reviewer_note`），業務方不知道具體哪一項要補。改成結構化：退件時勾選「哪幾項缺失」+ 文字說明，業務方補好後 resubmit，CTO 只需 review「改了什麼」。

### DB Schema 異動

加入 `submissions` table：
```python
# 退件結構化
rejection_reasons = Column(String, nullable=True)      # JSON array: ["dataset_count", "class_list"]
rejection_note = Column(String, nullable=True)         # 自由文字（取代 reviewer_note 中退件那部分）
resubmit_count = Column(Integer, default=0)           # 第幾次送審
resubmitted_at = Column(DateTime, nullable=True)      # 最近一次 resubmit 時間

# 原 reviewer_note 保留作為審核歷史（append-only）
```

新表：`submission_history`（審核軌跡）
```python
class SubmissionHistory(Base):
    __tablename__ = "submission_history"
    id = Column(Integer, primary_key=True)
    req_no = Column(String, index=True, nullable=False)
    action = Column(String, nullable=False)   # submit/approve/reject/resubmit
    actor = Column(String, nullable=False)    # CTO / HurricaneEdge-xxx
    reasons = Column(String, nullable=True)   # JSON
    note = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Why**：要看「這張工單走過哪幾輪退件」，就需要歷史紀錄。不想把舊 note overwrite。

### API 端點

新增 / 修改：
- `POST /api/submissions/{id}/reject` body: `{reasons: [...], note: "..."}`（取代現有 simple reject）
- `POST /api/submissions/{id}/resubmit` body: `{note: "修正了 X Y Z"}` → 狀態 submitted，resubmit_count +1
- `GET /api/submissions/{id}/history` → 回傳該工單的審核軌跡

### 前端 UX

1. **RejectModal 升級**：7 項 checklist 勾選的項目 → 存成 `rejection_reasons` JSON array
2. **SubmissionDetail 頁**：
   - 若 status=rejected → 顯示黃色警告框列出缺失項 + 「補件 resubmit」按鈕
   - 補件時彈 ResubmitModal，寫「改了什麼」
3. **ReviewPage**：若是 resubmit 的單，顯示「第 N 次送審」badge + diff（哪些欄位改過）
4. **SubmissionDetail** 新增「審核軌跡」時間軸 tab

### 驗收
- 業務方退件後，能看到具體哪幾項缺失
- 補件後再審，CTO 能一眼看到「改了什麼」
- 審核歷史完整保留

---

## Sprint 3 — Kaggle Kernel 狀態整合

### 業務目標
目前 ModelHub 不知道 Kaggle 訓練到哪一步，CEO 只能用 CLI 手動 poll。改成自動 poll + 顯示進度 + 完成後自動回填結果。

### DB Schema 異動

加入 `submissions` table：
```python
kaggle_kernel_slug = Column(String, nullable=True)      # boardgamegroup/mh-2026-xxx
kaggle_kernel_version = Column(Integer, nullable=True)
kaggle_status = Column(String, nullable=True)           # queued/running/complete/error
kaggle_status_updated_at = Column(DateTime, nullable=True)
kaggle_log_url = Column(String, nullable=True)          # 錯誤時連結 log
training_started_at = Column(DateTime, nullable=True)
training_completed_at = Column(DateTime, nullable=True)
```

### 實作：Background Poller

**Why poll 而不是 webhook**：Kaggle 沒有 webhook，只能 poll API。用 Python FastAPI 的 background task + APScheduler。

```python
# backend/pollers/kaggle_poller.py
# 每 60 秒 poll 一次所有 status=training 的 submission
# 呼叫 kaggle.api.kernels_status(slug)
# 狀態改變時寫 submission_history
```

Interval：60 秒（Kaggle rate limit ~100 req/hour，5 個 concurrent kernel 不會超）

### API 端點

- `POST /api/submissions/{id}/attach-kernel` body: `{slug: "...", version: N}` → 狀態 training，啟動 poll
- `GET /api/submissions/{id}/kaggle-status` → 回傳最新狀態
- `POST /api/submissions/{id}/refresh-kaggle` → 手動強制 poll 一次

### 前端 UX

1. **SubmissionDetail** 頁：status=training 時顯示：
   - Kaggle kernel badge（pending/running/complete/error）
   - 最後更新時間
   - 「手動刷新」按鈕
   - complete/error 時連結到 Kaggle log
2. **SubmissionListPage** 新增「訓練中」stat card 顯示 kernel 狀態分布
3. **ReviewPage** 通過時，若已知 kaggle_kernel_slug，自動綁上

### 驗收
- 審核通過 → attach kernel slug → 自動 poll
- 狀態變化 10 秒內反映到前端
- 訓練結束時，submission status 自動從 training → trained（等候 CEO/業務方驗收）

---

## Sprint 4 — 訓練結果自動回填 + 成本追蹤

### 業務目標
訓練完成時，現在要人工下載 log、抄 val_accuracy、寫 ModelVersion。改成自動：Kaggle 完成 → poll 下載 output → 解析指標 → 寫入 ModelVersion。同時追蹤每張工單的訓練成本（GPU 時數）。

### 自動回填機制

Poll 偵測 complete → 觸發：
1. `kaggle kernels output <slug> -p /tmp/mh-xxx` 下載 log
2. 正則解析 log 裡的 `val_accuracy / mAP50 / CER / final acc`
3. 建立 `ModelVersion` record，狀態 `pending_acceptance`
4. Submission 狀態 → `trained`

### 成本追蹤欄位

`submissions` 加：
```python
gpu_seconds = Column(Integer, nullable=True)
estimated_cost_usd = Column(Float, nullable=True)  # Kaggle GPU 免費，但折算 $0.5/hr P100 便於比較
total_attempts = Column(Integer, default=0)        # 第幾次訓練（失敗也算）
```

### 前端
- SubmissionList 顯示總 GPU 時數
- CEO 月報：本月累積訓練時數 + 估算成本

### 驗收
- 訓練完成 → 5 分鐘內 ModelVersion 自動出現
- CEO 能看本月累積 GPU 時數

---

## Sprint 5 — 通知 + 報表

### 業務目標
目前全靠手動盯狀態。加 CMC 通知 + 週報。

### 通知（透過 commtool CMC）

| 事件 | 通知誰 |
|------|--------|
| 新 submission | CTO |
| approved / rejected | submitter |
| training complete / failed | CTO + submitter |
| 訓練 > 24 小時未完成 | CTO（可能卡住）|

用 `notifications.py` 已有 hook，加 commtool CLI 呼叫即可。

### 週報

CEO 每週一早上收到：
- 本週新 submission / approved / rejected / trained
- 進行中訓練 + GPU 時數累積
- 超時工單（> 7 天未審）

寄到 `eric.wang@hurricanesoft.com.tw` + `anderson@hurricanecore.internal`

### 驗收
- 重要事件 1 分鐘內發 CMC
- 週一早上 09:00 自動送週報

---

## Sprint 4+ 未採納（但考慮過）

| 提案 | 評估 | 決定 |
|------|------|------|
| 部署追蹤（trained → deployed） | HurricaneEdge 自管部署，ModelHub 只管訓練 | **不做** |
| 權限系統（RBAC） | 目前 5 人使用，單一 LIDS 租戶即可 | **不做** |
| 模型 A/B test 支援 | 超出 ModelHub 職責 | **不做** |
| Dataset 版本控管 | Kaggle 自帶 | **不做** |
| 多雲訓練（Cloud Run / Lambda） | Kaggle 免費額度夠用，無需多雲 | **不做** |

**Why 不做**：工具要解決「當下痛點」，不要設計未來可能用到的功能。三個月後若真有需求再補。

---

## 實作順序建議

**第一波**（本週）：Sprint 2（結構化退件）— 每天都在用，效益最高
**第二波**（下週）：Sprint 3（Kaggle 整合）— CEO 不用再手動 poll
**第三波**（下下週）：Sprint 4 + 5 — 自動化 + 通知，長期省力

---

## 風險與擋路

1. **Kaggle API rate limit**：5+ concurrent poll 可能撞上限 → 用 exponential backoff
2. **SQLite 在 production**：目前 SQLite 沒問題（單機、低流量），但 Sprint 3 poll 高頻寫入要測 lock contention。若撞到改 PostgreSQL
3. **log 解析正則脆弱**：每個 kernel 的輸出格式不同（YOLO / CLS / OCR），要維護一個 parser registry

---

## 結論

**總工時 8 天**、**風險可控**、**每 Sprint 可獨立 deploy**。建議派 CTO 開發，分三波進，每 Sprint 完成 CEO 驗收一次。Sprint 2 是最大痛點（每次退件都要重寫 note），優先做。
