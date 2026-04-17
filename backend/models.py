import os
from datetime import datetime
from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# P3-1: 從 env 讀 DATABASE_URL；未設則用 SQLite（向後相容）
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/data/modelhub.db")

# SQLite 需要 check_same_thread=False；PostgreSQL 不需要
_is_sqlite = DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Submission(Base):
    """訓練需求單"""

    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    req_no = Column(String, unique=True, index=True, nullable=False)  # MH-2026-001
    # --- 基本資訊 ---
    req_name = Column(String, nullable=True)
    product = Column(String, nullable=False)
    company = Column(String, nullable=False)
    submitter = Column(String, nullable=True)
    purpose = Column(String, nullable=True)
    priority = Column(String, nullable=False, default="P2")
    # --- 模型規格 ---
    model_type = Column(String, nullable=True)
    class_list = Column(String, nullable=True)
    map50_threshold = Column(Float, nullable=True)
    map50_target = Column(Float, nullable=True)
    map50_95_target = Column(Float, nullable=True)
    inference_latency_ms = Column(Integer, nullable=True)
    model_size_limit_mb = Column(Integer, nullable=True)
    arch = Column(String, nullable=True, default="yolov8m")
    # --- 資料集 ---
    input_spec = Column(String, nullable=True)
    deploy_env = Column(String, nullable=True)
    dataset_source = Column(String, nullable=True)
    dataset_count = Column(String, nullable=True)
    dataset_val_count = Column(Integer, nullable=True)
    dataset_test_count = Column(Integer, nullable=True)
    class_count = Column(Integer, nullable=True)
    label_format = Column(String, nullable=True)
    kaggle_dataset_url = Column(String, nullable=True)
    dataset_path = Column(String, nullable=True)
    dataset_train_count = Column(Integer, nullable=True)
    # Sprint 10: 訓練產出的最佳模型路徑（取代借用 dataset_path 的臨時方案）
    model_output_path = Column(String, nullable=True)
    # --- 時程 ---
    expected_delivery = Column(String, nullable=True)
    # --- 狀態機 ---
    status = Column(String, nullable=False, default="draft")
    # --- 審核 ---
    reviewer_note = Column(String, nullable=True)
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    # --- Sprint 2: 結構化退件 ---
    rejection_reasons = Column(String, nullable=True)      # JSON array string
    rejection_note = Column(String, nullable=True)
    resubmit_count = Column(Integer, default=0)
    resubmitted_at = Column(DateTime, nullable=True)
    # --- Sprint 3: Kaggle Kernel 整合 ---
    kaggle_kernel_slug = Column(String, nullable=True)     # username/kernel-name
    kaggle_kernel_version = Column(Integer, nullable=True)
    kaggle_status = Column(String, nullable=True)          # queued/running/complete/error
    kaggle_status_updated_at = Column(DateTime, nullable=True)
    kaggle_log_url = Column(String, nullable=True)
    training_started_at = Column(DateTime, nullable=True)
    training_completed_at = Column(DateTime, nullable=True)
    # --- Sprint 4: 訓練結果 + 成本 ---
    gpu_seconds = Column(Integer, nullable=True)
    estimated_cost_usd = Column(Float, nullable=True)
    total_attempts = Column(Integer, default=0)
    # --- Sprint 6: 自動重試 + 預算 ---
    max_retries = Column(Integer, default=2)
    retry_count = Column(Integer, default=0)
    max_budget_usd = Column(Float, default=5.0)
    budget_exceeded_notified = Column(Boolean, default=False)
    # --- 資料集狀態（dataset unblock）---
    dataset_status = Column(String, nullable=False, default="ready")  # ready/missing_labels/missing_data/partial
    blocked_reason = Column(String, nullable=True)
    # --- Sprint 13 P2-A: per-class metrics（JSON string, {class_name: ap50}）---
    per_class_metrics = Column(String, nullable=True)
    # --- 時間戳記 ---
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    """模型版本清冊"""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    req_no = Column(String, index=True, nullable=False)
    product = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    train_date = Column(String, nullable=True)
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    file_path = Column(String, nullable=True)
    status = Column(String, nullable=False, default="active")
    notes = Column(String, nullable=True)
    kaggle_kernel_url = Column(String, nullable=True)
    epochs = Column(Integer, nullable=True)
    batch_size = Column(Integer, nullable=True)
    arch = Column(String, nullable=True)
    map50_actual = Column(Float, nullable=True)
    map50_95_actual = Column(Float, nullable=True)
    pass_fail = Column(String, nullable=True)
    accepted_by = Column(String, nullable=True)
    accepted_at = Column(DateTime, nullable=True)
    acceptance_note = Column(String, nullable=True)
    is_current = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ApiKey(Base):
    """Sprint 7.1 — DB-backed API key 管理"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)                 # 用途標籤
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    disabled = Column(Boolean, default=False)


class SubmissionHistory(Base):
    """審核軌跡（append-only）— Sprint 2"""

    __tablename__ = "submission_history"

    id = Column(Integer, primary_key=True, index=True)
    req_no = Column(String, index=True, nullable=False)
    action = Column(String, nullable=False)       # submit/approve/reject/resubmit/attach_kernel/complete/...
    actor = Column(String, nullable=True)
    reasons = Column(String, nullable=True)       # JSON array string
    note = Column(String, nullable=True)
    meta = Column(String, nullable=True)          # JSON blob 任意附加資料
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# SQLite 手動 migration：新欄位補到既有 table
_MIGRATIONS = [
    # (table, column, ddl_suffix)
    ("submissions", "rejection_reasons",        "VARCHAR"),
    ("submissions", "rejection_note",           "VARCHAR"),
    ("submissions", "resubmit_count",           "INTEGER DEFAULT 0"),
    ("submissions", "resubmitted_at",           "DATETIME"),
    ("submissions", "kaggle_kernel_slug",       "VARCHAR"),
    ("submissions", "kaggle_kernel_version",    "INTEGER"),
    ("submissions", "kaggle_status",            "VARCHAR"),
    ("submissions", "kaggle_status_updated_at", "DATETIME"),
    ("submissions", "kaggle_log_url",           "VARCHAR"),
    ("submissions", "training_started_at",      "DATETIME"),
    ("submissions", "training_completed_at",    "DATETIME"),
    ("submissions", "gpu_seconds",              "INTEGER"),
    ("submissions", "estimated_cost_usd",       "FLOAT"),
    ("submissions", "total_attempts",           "INTEGER DEFAULT 0"),
    ("submissions", "max_retries",              "INTEGER DEFAULT 2"),
    ("submissions", "retry_count",              "INTEGER DEFAULT 0"),
    ("submissions", "max_budget_usd",           "FLOAT DEFAULT 5.0"),
    ("submissions", "budget_exceeded_notified", "BOOLEAN DEFAULT FALSE"),
    ("model_versions", "is_current",            "BOOLEAN DEFAULT FALSE"),
    # Sprint 7.1 — api_keys table 自動建，ALTER 只補缺欄位時用
    # Dataset unblock — 資料集狀態欄位
    ("submissions", "dataset_status",           "VARCHAR DEFAULT 'ready'"),
    ("submissions", "blocked_reason",           "TEXT"),
    # Sprint 10: 訓練產出路徑（取代 dataset_path 借用方案）
    ("submissions", "model_output_path",        "TEXT"),
    # Sprint 13 P2-A: per-class AP50 metrics（JSON string）
    ("submissions", "per_class_metrics",        "TEXT"),
]


_DATASET_UNBLOCK_SEED = [
    (
        "MH-2026-005",
        "missing_labels",
        "270 張原始圖片已備齊，但缺 YOLO 標籤檔（.txt）。可用 010 訓練好的 PID 模型 auto-label。",
    ),
    (
        "MH-2026-006",
        "missing_labels",
        "360 張原始圖片已備齊，但缺 YOLO bbox 標籤。可用 PaddleOCR/EasyOCR auto-detect 文字框產生。",
    ),
    (
        "MH-2026-007",
        "missing_data",
        "只有 2 張原始圖嚴重不足，且缺 segmentation mask。需先合成或補齊到 ≥ 200 張，再用 Canny+Hough 自動產 mask。",
    ),
    (
        "MH-2026-008",
        "missing_labels",
        "212 張原始圖片已備齊，但缺 YOLO bbox 標籤。可用 PID 模型 + 邊界規則 auto-label。",
    ),
]


def init_db():
    if _is_sqlite:
        # SQLite: 保留原有 create_all + ALTER TABLE 手動 migration（向後相容）
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            for table, column, ddl in _MIGRATIONS:
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                    conn.commit()
                except Exception:
                    pass  # 欄位已存在，忽略
        # 填入工單阻塞原因（idempotent：只在欄位為空時寫入，不覆蓋已有內容）
        with engine.connect() as conn:
            for req_no, ds_status, reason in _DATASET_UNBLOCK_SEED:
                conn.execute(
                    text(
                        "UPDATE submissions SET dataset_status = :ds, blocked_reason = :r "
                        "WHERE req_no = :rn AND (blocked_reason IS NULL OR blocked_reason = '')"
                    ),
                    {"ds": ds_status, "r": reason, "rn": req_no},
                )
            conn.commit()
    else:
        # PostgreSQL: 走 alembic upgrade head
        import subprocess
        import sys
        import os as _os
        alembic_ini = _os.path.join(_os.path.dirname(__file__), "alembic.ini")
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", alembic_ini, "upgrade", "head"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            import logging
            logging.getLogger("modelhub.models").error(
                "alembic upgrade head failed: %s", result.stderr
            )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
