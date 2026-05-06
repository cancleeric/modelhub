# CLAUDE.md — HurricaneCore（颶核科技）工作區

## 🎨 UI 優化工作流（2026-04-26 新增 — 強制）

**任何「優化 UI」「改介面」「重設計頁面」任務一律走 Chrome MCP + claude.ai/design，禁止閉門造車改 CSS。**

標準流程：
1. **Phase 1 調研**：派 CPO 副手啟本機服務（不動 Docker）、列頁面清單、抓現有設計 token、列 UI 痛點清單
2. **Phase 2 截圖**：CEO 用 Chrome MCP（`mcp__claude-in-chrome__*`）開瀏覽器截關鍵頁
3. **Phase 3 設計簡報**：寫到 `/tmp/ui-design-brief-<date>.md`，含產品定位 + 痛點清單 + 截圖 ID + design token 提案 + 期望輸出
4. **Phase 4 出設計稿**：老闆貼簡報到 https://claude.ai/design，要求 React + Tailwind 可 ship 程式碼
5. **Phase 5 實作驗收**：派 CTO 副手依新稿實作，Chrome MCP 截 A/B 對照

例外：1 行樣式微調可不走流程；只要碰到「整頁重設計/視覺改造」必走。

完整規則：`~/.claude/rules.md` UI 優化工作流。

---

## 🖥️ ModelHub 訓練資源規範（強制）

**訓練任務必須按優先序使用資源，不得直接跑本機：**

1. **免費雲端 GPU（優先）**
   - Kaggle：帳號 `boardgamegroup`，T4/P100，**30 hr/週免費配額**
   - 每個訓練需求單開訓前，必須先確認是否有 Kaggle kernel（`kaggle-kernels/` 目錄）
   - 若無 kernel → 先建 kernel + 上傳 dataset，再送 Kaggle 跑
   - Kaggle 配額耗盡才考慮下一層
2. **其他免費資源**（Kaggle 不可用時）
   - Lightning AI（22 hr/月）、SSH 內網主機（先確認有 CUDA GPU）
3. **本機 MPS（最後手段）**
   - 本機不是專用訓練機，同時只能跑 **一個** 訓練任務
   - 本機訓練要用 `--device mps`，禁止 `device="cpu"` 寫死
   - 多個任務用隊列排序（`training_queue.sh`），不得並行

**違反此規範 = 浪費免費資源、燒本機效能。**

---

## ⛔ DB 存取規範（集團強制）

生產 DB 異動必須遵守集團規範：
📄 [集團 DB 存取政策](../HurricaneGroup/docs/shared/db-access-policy.md)

**核心規則**：
- 禁止直接 psql 連生產 DB
- 密碼只從 GCP Secret Manager（生產）或 Hurricane Vault（開發）取得
- Schema 變更必須走 migration（不可手動 SQL）
- L3-L4 異動需老闆核准

## 🔐 Hurricane Vault（開發環境密碼）

本機開發的密碼改從 Hurricane Vault 取得：
- Server：`http://localhost:8930`
- 使用手冊：[dev onboarding](../HurricaneGroup/docs/shared/runbooks/hurricane-vault-dev-onboarding.md)
- 命名規範：`hurricanecore/dev/<KEY>`、`anemone/dev/<KEY>`、`brain/dev/<KEY>`
- 啟動服務建議用 wrapper：`hvault run --prefix "anemone/dev/" -- <cmd>`

---

> 工作區路徑：`~/HurricaneCore/`
> 公司：HurricaneCore（颶核科技），颶風集團第六家子公司
> 定位：**集團 AI 核心引擎**（不做業務邏輯，只做 AI 能力輸出）
> CEO：**Anderson**（2026-04-05 就任）
> 成立日期：2026-04-05
> LIDS 租戶：`hurricanecore` / id `8fb87999-a5ec-4964-bbb3-1cb5cd49e729`（Active）
> CEO 帳號：`ceo@hurricanecore.internal`（密碼在 Vault `hurricanecore/dev/CEO_LIDS_PASSWORD`）
> Mail domain：`hurricanecore.internal`（Stalwart 本地投遞，不走外部 DNS）
> 最後更新：2026-04-20

---

## 公司定位

颶核科技**不做業務邏輯**（排盤、訂位、記帳、遊戲）。
颶核科技做的是：**讓 AI 能力流入每一個產品**。

```
其他公司做業務 → 颶核科技供 AI 能力 → Widget 出現在客戶面前
```

## 產品線

| Repo | 產品 | 說明 | Dev Port | 狀態 |
|------|------|------|----------|------|
| `anemone-platform/` | Anemone（海葵）| 即時客服中繼（Widget ↔ Server ↔ OpenClaw）| 8920 (server) / 3920 (widget) | 原 HurricaneSoft 研究項目，轉入颶核 |
| `brain/` | 颶核智腦平台 | 集團 LLM Gateway + 計費中心（Gemini Headless 預設 / Claude PTY / DeepSeek）| TBD | 待建 |
| `wiki/` | **Wiki.js v2**（颶核維護的集團技術文檔平台）| 集團跨公司 Markdown wiki，LIDS SSO，含 13 group 權限隔離 | 8933 (app)，DB 已併入 squid-postgres-dev | 運行中 |

### Anemone 架構
```
Widget (客戶端 JS)
    │
    ▼
Anemone Server (FastAPI 中繼，本機 dev / 暫停 Cloud Run)  ← stateless，禁止直連 DB
    ↑  WS（Brain Cloud 主動連出 /ws/brain）
Brain Cloud (多租戶管理層，本機 brain-cloud-dev:8932 / 暫停 Cloud Run)
    ↑  WS（brain-llm-connector 主動連出 /ws/llm-provider）
Brain API (LLM Gateway，本機 brain-api-dev:8931)
    │
    ▼
LLM Provider（Gemini Headless 預設）
```
**連線方向：全部由下層主動連出，上層接受。**
**三層分離**：Widget 只做 UI；Anemone Server 只做中繼、不做 AI；Brain Cloud/Brain API 才是智能層。

> ⚠️ **2026-04-20 狀態**：GCP Cloud Run（Anemone + Brain Cloud）已暫停，目前全部在本機開發。
> `brain-llm-connector-prod` 的 `BRAIN_CLOUD_WS_URL` 仍指向舊 GCP URL（已失效），恢復雲端前需更新。
> 工單 #202604200002（CTO）：connector 自動重連邏輯，雲端恢復前為低優先。

### DB 架構規範（2026-04-17 老闆拍板）

> 智腦平台只用一個 DB：`brain_db`（由 brain-cloud 管理）

- **Anemone Server 禁止直連任何 DB**（stateless 中繼，不持有 DB 連線）
- Anemone 需要 persistence 時，改呼叫 brain-cloud HTTP API
- **所有 domain 資料歸 brain_db**：conversations、messages、widget_configs、scores、feedback、review_queue
- **新 table 都進 brain_db**，未經老闆同意不得另建新 DB
- anemone_db 已廢除完成（2026-04-17，`anemone-db-dev` 容器已不存在）

### 智腦員工規劃

| 智腦 | 服務產品 | 服務 API | 狀態 |
|------|---------|----------|------|
| 天機智腦 | 天機（命理）| tianji-api :8300 | 待建 |
| 訂位智腦 | RS Conch（訂位）| rs-api :8100 | 待建 |
| CAD 智腦 | AICAD | aicad-api :8200 | 待建 |
| 記帳智腦 | 算盤 Abacus | abacus-api :5101 | 待建 |
| 打工仔智腦 | 打工仔 | dagongzai-api :8002 | 待建 |
| 官網客服 | 颶風軟體官網 | 各產品 API | 待建 |

---

## C-Suite（本地 agents）

虛擬角色定義在 `.claude/agents/`，由 Claude Code 自動載入，可用 Agent tool 呼叫：

| 角色 | 檔案 | 職責 |
|------|------|------|
| CEO | `ceo.md` | 公司治理、向 Eric 負責 |
| CTO | `cto.md` | 技術架構、Code Review、容器維運 |
| COO | `coo.md` | 跨專案統籌、發版協調 |
| CPO | `cpo.md` | 產品規劃、測試策略 |
| CBO | `cbo.md` | 商務、定價、成本控管 |
| CISO | `ciso.md` | 資安政策、合規審查 |
| CoS | `cos.md` | 幕僚長、協調副總、會議紀錄 |
| Legal | `legal.md` | 合約、服務條款、法規遵循 |

⚠️ 颶核科技的 C-Suite **不指揮** Eric 的直屬員工（Samantha、Estella 等），只能用自己公司內的虛擬角色。

---

## 版控

| Repo | Gitea Org | 狀態 |
|------|-----------|------|
| `anemone-platform` | `hurricanesoft/anemone-platform` → 待遷移至 `hurricanecore` | ✅ 主 repo（暫放 HurricaneSoft org）|
| `brain` | `hurricanecore/brain` | 待建 |
| ~~`anemone`~~ | `hurricanesoft/anemone` | 🗄 封存 |
| ~~`anemone-widget`~~ | `hurricanesoft/anemone-widget` | 🗄 封存 |

**SSH Remote**：`ssh://git@localhost:2230/hurricanesoft/{repo}.git`（Gitea 本機）

---

## Docker 容器管制

> **管制文件**：`~/HurricaneSoft/infra-private/Docker-Port-管理文件.md`（Estella 管理）
> **⛔ 不可自行分配 Port、改容器名、建/改 Network** — 需依流程申請

### 命名規範
```
{project}-{service}-{env}    全小寫、用 - 連接、禁止底線 _ 或大寫
```

### HurricaneCore 已分配 Port

| Port | 容器名 | 服務 |
|------|--------|------|
| 3920 | `anemone-widget-dev` | Anemone Widget（客戶端 Demo）|
| 3931 | `brain-console-dev` | Brain Console（Next.js 後台 UI）|
| 3950 | `modelhub-web-dev` | ModelHub Frontend（Vite + React）|
| 5440 | `anemone-db-dev` | Anemone DB（已停用，2026-04-17 併入 brain_db）|
| 5441 | `brain-cloud-db-dev` | Brain Cloud PostgreSQL（brain_db + pgvector 向量 DB 專用，未來有向量需求的 DB 一律放這）|
| 8920 | `anemone-server-dev` | Anemone Server（FastAPI 中繼）|
| 8931 | `brain-api-dev` | Brain API（LLM Gateway + 計費中心）|
| 8932 | `brain-cloud-dev` | Brain Cloud（智腦管理層）|
| 8950 | `modelhub-api-dev` | ModelHub Backend（FastAPI + SQLite）|
| 8941 | `aegis-daemon-dev` | Aegis v3 資安 Agent（standalone reporting API）|
| —    | `modelhub-connector-dev` | ModelHub LLM Connector（無對外 Port，WS 連出至 brain-cloud-dev:8932）|

> 已同步 Estella 管制文件 `~/HurricaneSoft/infra-private/Docker-Port-管理文件.md`（2026-04-26 補登記；8941 於 2026-04-27 新增；modelhub-connector-dev 於 2026-05-04 新增）

新增服務向 Estella 申請。

### Network
統一連 `squid_dev_network`（172.20.0.0/16），不建立專案獨立 network。

---

## 集團架構（參考）

| 公司 | 英文名 | CEO | 工作區 |
|------|--------|-----|--------|
| 颶風軟體 | HurricaneSoft | Eric | `~/HurricaneSoft/` |
| 颶風科技 | HurricaneTech | Kay | `~/HurricaneTech/` |
| 颶鋒科技 | HurricaneEdge | Rachel | `~/HurricaneEdge/` |
| 颶電娛樂 | HurricaneDigital | Nicholas | `~/HurricaneDigital/` |
| 颶擎科技 | HurricanePrime | Lucas | `~/HurricanePrime/` |
| **颶核科技** | **HurricaneCore** | **待定** | **`~/HurricaneCore/`** |

---

## 跨公司分工

| 職責 | 負責公司 | 備註 |
|------|---------|------|
| AI 能力輸出（Anemone/颶核智腦）| **HurricaneCore**（你）| 本公司主業 |
| 集團技術文檔平台 Wiki.js | **HurricaneCore**（你）| CEO Anderson 負責部署維運、權限結構、內容規範 |
| OpenClaw 引擎 | HurricaneSoft | 智腦管理層之一（與 Brain Cloud 同層），LLM 透過 brain 取得，不直連 |
| LobsterFarm / LIDS / CMS | HurricaneTech | 智腦員工管理平台 |
| 業務產品（天機/RS/AICAD/算盤）| HurricaneEdge | 提供 API 給智腦呼叫 |
| 遊戲產品 | HurricaneDigital | 未來接遊戲客服智腦 |
| 客製專案 | HurricanePrime | 客戶 SaaS 可配專屬智腦 |

---

## 技術文件系統

**集團統一閱讀入口 — Wiki.js**（2026-04-11 老闆決策）：

- **閱讀入口**：http://localhost:8933（LIDS SSO 登入）
- **Source of truth**：`~/HurricaneGroup/docs/` → Gitea `hurricanesoft/docs.git`
- **同步機制**：短期靠批次腳本（WIKI-004 已遷入 1,322 頁），長期靠 Wiki.js Git Storage 雙向同步（WIKI-006）

### 主要 Wiki 路徑

- 集團總入口：http://localhost:8933/home
- 集團共用文件：http://localhost:8933/shared/*
- 颶核科技文件：http://localhost:8933/hurricanecore/*
  - brain（LLM Gateway + 計費中心）：http://localhost:8933/hurricanecore/brain/*
  - Anemone 電話總機：http://localhost:8933/hurricanecore/anemone/*
  - brain-cloud 智腦管理層：http://localhost:8933/hurricanecore/brain-cloud
  - runbooks：http://localhost:8933/hurricanecore/runbooks

### Gitea 原始位置（source of truth，寫文件在這）

- 集團根索引：`~/HurricaneGroup/docs/README.md`
- 颶核科技文件：`~/HurricaneGroup/docs/hurricanecore/`
- Postmortems：`~/HurricaneGroup/docs/shared/postmortems/`

**原則**：**寫** 文件在 `~/HurricaneGroup/docs/`，**讀** 文件在 Wiki.js。GCP 內容禁入 Wiki（由 HurricaneSoft 統一維護）。

---

## 待辦

| 項目 | 負責 | 狀態 |
|------|------|------|
| CEO 人選確認 | Eric | ✅ Anderson (2026-04-05) |
| LIDS 建立 `hurricanecore` 租戶 | Eric | ✅ 2026-04-05 |
| Anderson LIDS + Mail + CMS 帳號 | Eric | ✅ 2026-04-05 |
| Stalwart `hurricanecore.internal` domain | Eric | ✅ 2026-04-05 |
| Gitea 建立 `hurricanecore` org | Eric | ✅ 2026-04-07 |
| Anemone repo 遷移至 hurricanecore org | CTO | ✅ 2026-04-07 |
| 第一個智腦（天機）建立 | 颶核 + HurricaneEdge | ⏳ 待執行 |
| stalwart-hs fork + LIDS 整合 | CTO（2026-04-05 派發背景） | 🏃 進行中 |
| Vault 審計/版本補到 GCP parity | CTO（2026-04-05 派發背景） | 🏃 進行中 |

## 2026-04-05 首日事件摘要

踩到的架構漏洞（已開 issue，改造中）：
1. LIDS rate limit 文檔缺口 → `hurricanetech/LocalIdentityServer#101`
2. LIDS SuperAdmin 密碼變更需同步 Vault → `hurricanetech/LocalIdentityServer#102`
3. Stalwart 未接 LIDS SSO + 沒 fork → `hurricanetech/commtool-server#93`
4. Stalwart `docker restart` 會觸發 entrypoint regen config — 未來改用 `docker kill --signal=HUP`

完整記錄：[shared/announcements/hurricanecore-launch-20260405.md](../HurricaneGroup/docs/shared/announcements/hurricanecore-launch-20260405.md)

---

## 注意事項

- 颶核科技獨立運作，**不指揮**其他公司員工
- 智腦呼叫業務 API 要走 LIDS 認證、tenant 隔離
- 所有系統禁止直連 LLM 廠商 API，統一走 brain（集團 LLM Gateway，預設 Gemini Headless）
- 生產部署找 HurricaneSoft（Estella）
- LobsterFarm/LIDS/CMS 問題找 HurricaneTech（Kay）
- 工單流程：neritic 開單 → feature branch → PR → 送審
- **Wiki.js 集團技術文檔平台由颶核維護**：http://localhost:8933，LIDS SSO 登入，路徑規範 `/<tenant>/...`，GCP 內容禁入（由 HurricaneSoft 統一維護）

---

## 集團技術文檔 Wiki（2026-04-11 新規）

### Wiki.js 規範

集團統一使用 **Wiki.js** 管理技術文檔：

- **URL**：http://localhost:8933
- **登入**：LIDS SSO（點登入頁的「LIDS SSO」按鈕，用你的 LIDS 帳號登入）
- **維護**：颶核科技 CEO Anderson（wiki 本身的部署與維運）
- **路徑規範**：`<tenant_slug>/<category>/<page>`，例：
  - `/hurricanesoft/...` — 颶風軟體文件
  - `/hurricanecore/...` — 颶核科技文件
  - `/hurricanetech/...` — 颶風科技文件
  - `/hurricaneedge/...` — 颶鋒科技文件
  - `/hurricanedigital/...` — 颶電娛樂文件
  - `/hurricaneprime/...` — 颶擎科技文件
  - `/shared/...` — 集團共用內容（公告、ADR、runbooks、policies）

### 技術文檔必須寫入 Wiki

所有**正式**技術文檔（架構、部署、runbook、API 規格、ADR）**必須**進 Wiki，不得只放在 repo 的 `docs/` 而不同步。

同步方式：
1. **短期**：寫在 `docs/` 底下的 md 檔會被批次腳本遷入 Wiki（見 neritic Brain **Wiki** project）
2. **長期**：Wiki.js Git Storage 雙向同步到 Gitea（neritic **#202604110008** 進行中）

### ⛔ 禁止寫入 Wiki 的內容

1. **GCP Cloud Run 相關的設定、URL、project id、service account key** — **由 HurricaneSoft 統一維護**，不寫入 Wiki
2. **任何密碼 / API key / token / private key** — 一律放 Hurricane Vault
3. **個人資料（PII）** — 客戶、員工的私人資訊
4. **合約、法務敏感文件** — 僅限 legal 角色看

Wiki 的遷入腳本有 secret pattern 過濾（密碼、AWS key、GitHub token、RSA private key 等），但**人的責任是第一道防線**，請在寫 md 時就不要放敏感內容。

### GCP 分工（2026-04-11 新規）

**GCP Cloud Run 及相關帳密由 HurricaneSoft 統一維護**，其他子公司不直接操作：
- 部署到 Cloud Run 找 HurricaneSoft Estella
- GCP project id / service account / API key 不分享到 Wiki 或其他 repo
- 需要 GCP 資源請開 issue 給 HurricaneSoft

---
