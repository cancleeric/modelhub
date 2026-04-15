# ModelHub — ML 模型訓練需求與版本管理系統

颶核科技 ModelHub 提供：
- 模型訓練需求單管理（Submissions）
- 模型版本清冊（Registry）

## 快速啟動

```bash
# 啟動服務（port 8950，暫用，待 Estella 正式分配）
docker compose up -d

# API 文件
open http://localhost:8950/docs

# 執行 Phase 0 種子資料
docker exec modelhub-api-dev python seed_data.py
```

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
