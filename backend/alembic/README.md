# Alembic Migration 說明

## 新增 Migration 流程

所有 schema 變更（新表、新欄位、索引調整）統一走 Alembic，**不要**在 `models.py` 的 `_MIGRATIONS` 清單手動加 ALTER TABLE。

```bash
# 1. 在 models.py 的 SQLAlchemy model 加好新欄位
# 2. 自動產生 migration
alembic revision --autogenerate -m "描述你的變更"
# 3. 確認產出的 migration file 是否正確
# 4. 套用
alembic upgrade head
```

## 環境說明

| 環境 | DB | Migration 方式 |
|------|----|---------------|
| 本機 dev（SQLite）| `data/modelhub.db` | `init_db()` 執行 `_MIGRATIONS`（向後相容）+ Alembic 可選 |
| Docker / 生產（PostgreSQL）| `DATABASE_URL` 指向 PG | `init_db()` 自動執行 `alembic upgrade head` |

## 注意事項

- `models.py` 的 `_MIGRATIONS` 是歷史遺留清單，不再新增。
- PostgreSQL 路徑下 `_MIGRATIONS` 完全不執行，只走 Alembic。
- downgrade 未必安全，請在執行前備份 DB。

## 既有 Revision 清單

| Revision | 說明 |
|----------|------|
| `0001_initial` | 初始 schema（submissions / model_versions / api_keys / submission_history）|
| `0002_training_queue` | 新增 training_queue 表 |
