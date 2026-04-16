# ModelHub 跨公司整合指南

適用對象：各子公司（HurricaneEdge、HurricanePrime 等）需要從 ModelHub 拉取 AI 模型的工程師。

## 概覽

ModelHub 提供兩個機器對機器（M2M）API，使用 **API Key** 認證，不需要 LIDS SSO：

| API | 說明 |
|-----|------|
| `GET /api/registry/latest` | 取得指定 product + model_name 的最新通過版本 |
| `GET /api/registry/{id}/download` | 下載模型檔案 |

## 認證

所有 M2M 請求須帶 `X-Api-Key` header：

```http
X-Api-Key: <MODELHUB_API_KEY>
```

API Key 從 Hurricane Vault 取得：

```bash
hvault get hurricanecore/dev/MODELHUB_API_KEY --show -q
```

若無 Vault 存取權限，向 HurricaneCore CTO 申請。

## API 說明

### GET /api/registry/latest

取得最新的 `is_current=true` 且 `pass_fail=pass` 版本。

**Parameters**

| 參數 | 必填 | 說明 |
|------|------|------|
| `product` | 是 | 產品代碼，如 `AICAD`、`tianji` |
| `model_name` | 是 | 模型名稱，如 `pid`、`instrument_yolo` |

**Example**

```bash
curl -H "X-Api-Key: $MODELHUB_API_KEY" \
  "http://modelhub.hurricanecore.internal:8950/api/registry/latest?product=AICAD&model_name=pid"
```

**Response**

```json
{
  "id": 1,
  "req_no": "MH-2026-001",
  "product": "AICAD",
  "model_name": "pid",
  "version": "v2",
  "map50_actual": 0.923,
  "pass_fail": "pass",
  "is_current": true,
  "accepted_at": "2026-04-15T10:00:00",
  "file_path": "pid_model_v2_20260415.pt"
}
```

### GET /api/registry/{id}/download

下載模型檔案（binary）。

**Example**

```bash
curl -H "X-Api-Key: $MODELHUB_API_KEY" \
  "http://modelhub.hurricanecore.internal:8950/api/registry/1/download" \
  -o pid_model.pt
```

## 推薦整合方式

使用 `docs/examples/pull_latest_model.py` 腳本，支援：
- 自動比對版本，未更新時跳過下載
- 本地 cache 紀錄（`.modelhub_cache.json`）
- 命令列參數指定 product、model、output 路徑

```bash
python pull_latest_model.py \
  --product AICAD \
  --model pid \
  --output ./models/pid_model.pt
```

環境變數：

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `MODELHUB_URL` | `http://localhost:8950` | ModelHub 服務 URL |
| `MODELHUB_API_KEY` | （需設定）| API Key |

## CI/CD 整合建議

在 CI pipeline 的 pre-deploy 步驟加入：

```yaml
- name: Pull latest model
  env:
    MODELHUB_URL: http://modelhub.hurricanecore.internal:8950
    MODELHUB_API_KEY: ${{ secrets.MODELHUB_API_KEY }}
  run: |
    python scripts/pull_latest_model.py \
      --product AICAD \
      --model pid \
      --output app/models/pid_model.pt
```

## 聯絡

ModelHub 由 **HurricaneCore** 維護。問題請開 issue 至 `hurricanecore/modelhub` 或聯絡 HurricaneCore CTO。
