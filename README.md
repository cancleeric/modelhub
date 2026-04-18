# ModelHub — ML 模型訓練需求與版本管理系統

颶核科技 ModelHub 提供：
- 模型訓練需求單管理（Submissions）
- 模型版本清冊（Registry）

## ⚠️ 訓練資源使用規範

**訓練任務必須按以下優先序，不得直接跑本機：**

| 優先級 | 資源 | 額度 | 條件 |
|--------|------|------|------|
| 1 | **Kaggle**（boardgamegroup）| T4/P100，30 hr/週免費 | 需有 `kaggle-kernels/` 對應目錄 |
| 2 | Lightning AI / 內網 SSH GPU | 22 hr/月 / 視硬體 | Kaggle 配額耗盡時 |
| 3 | **本機 MPS**（最後手段）| 不限 | 前兩者不可用時 |

**本機訓練規則：**
- 同時只能跑 **一個** 任務（隊列排序）
- 必須用 `--device mps`，禁止 `device="cpu"` 寫死
- 不得獨佔全部資源，其他服務要能正常運行

新需求單開訓前，先確認 `kaggle-kernels/` 有無對應 kernel。沒有 → 先建 kernel → 上傳 dataset → 送 Kaggle。

---

## 快速啟動

```bash
# 啟動服務（推薦方式：自動從 Hurricane Vault 注入 API key）
./start.sh

# 傳遞額外 docker-compose 參數（例如重建 image）
./start.sh --build

# 手動啟動（需自行設定環境變數）
docker compose up -d

# API 文件
open http://localhost:8950/docs

# 執行 Phase 0 種子資料
docker exec modelhub-api-dev python seed_data.py
```

> `start.sh` 會自動呼叫 `hvault run --prefix "hurricanecore/dev/"` 注入
> MODELHUB_API_KEY、LIGHTNING_API_KEY 等 Vault 機密，並允許透過
> `DATABASE_URL` 環境變數覆寫資料庫位置（預設 SQLite）。

## 專案結構

```
modelhub/
├── backend/
│   ├── main.py              FastAPI app 入口
│   ├── models.py            SQLite DB schema（SQLAlchemy）
│   ├── seed_data.py         Phase 0 初始資料種子
│   ├── routers/
│   │   ├── submissions.py   需求單 CRUD
│   │   └── registry.py      模型版本清冊 CRUD
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                Phase 1 再建
├── docs/                    指向 Wiki 及 docs repo
└── docker-compose.yml
```

## API Endpoints

| Method | Path | 說明 |
|--------|------|------|
| GET | /health | 健康檢查 |
| GET | /api/submissions/ | 列出需求單 |
| POST | /api/submissions/ | 建立需求單 |
| GET | /api/submissions/{req_no} | 取得需求單 |
| PATCH | /api/submissions/{req_no} | 更新需求單 |
| DELETE | /api/submissions/{req_no} | 刪除需求單 |
| GET | /api/registry/ | 列出模型版本 |
| POST | /api/registry/ | 建立模型版本 |
| GET | /api/registry/{id} | 取得模型版本 |
| PATCH | /api/registry/{id} | 更新模型版本 |
| DELETE | /api/registry/{id} | 刪除模型版本 |

## 技術文件

詳見 [docs/README.md](docs/README.md)
