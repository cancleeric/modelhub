from datetime import datetime
from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite:////app/data/modelhub.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Submission(Base):
    """訓練需求單"""

    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    req_no = Column(String, unique=True, index=True, nullable=False)  # MH-2026-001
    # --- 基本資訊 ---
    req_name = Column(String, nullable=True)            # 需求名稱
    product = Column(String, nullable=False)            # AICAD / 天機 / RS 等
    company = Column(String, nullable=False)            # 業務方公司，如 HurricaneEdge
    submitter = Column(String, nullable=True)           # 業務方聯絡人
    purpose = Column(String, nullable=True)             # 業務問題描述
    priority = Column(String, nullable=False, default="P2")  # P0/P1/P2/P3
    # --- 模型規格 ---
    model_type = Column(String, nullable=True)          # detection/classification/segmentation
    class_list = Column(String, nullable=True)          # 類別清單（逗號分隔）
    map50_threshold = Column(Float, nullable=True)      # 舊欄位，保留相容
    map50_target = Column(Float, nullable=True)         # mAP50 目標值（如 0.90）
    map50_95_target = Column(Float, nullable=True)
    inference_latency_ms = Column(Integer, nullable=True)   # 推論速度要求（ms）
    model_size_limit_mb = Column(Integer, nullable=True)
    arch = Column(String, nullable=True, default="yolov8m")  # 模型架構
    # --- 資料集 ---
    input_spec = Column(String, nullable=True)          # 輸入規格描述
    deploy_env = Column(String, nullable=True)          # 部署目標，如 aicad-api :8200
    dataset_source = Column(String, nullable=True)      # 訓練資料來源
    dataset_count = Column(String, nullable=True)       # 資料集大小描述（舊欄位）
    dataset_val_count = Column(Integer, nullable=True)
    dataset_test_count = Column(Integer, nullable=True)
    class_count = Column(Integer, nullable=True)
    label_format = Column(String, nullable=True)        # 標注格式，如 YOLO
    kaggle_dataset_url = Column(String, nullable=True)
    # --- 時程 ---
    expected_delivery = Column(String, nullable=True)   # 預計交付日期
    # --- 狀態機 ---
    # draft → submitted → approved/rejected → training → trained → accepted/failed
    status = Column(String, nullable=False, default="draft")
    # --- 審核 ---
    reviewer_note = Column(String, nullable=True)       # CTO 審核意見
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    # --- 時間戳記 ---
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    """模型版本清冊"""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    req_no = Column(String, index=True, nullable=False)    # 對應需求單
    product = Column(String, nullable=False)
    model_name = Column(String, nullable=False)            # PID 管線辨識
    version = Column(String, nullable=False)               # v2
    train_date = Column(String, nullable=True)             # 2026-04-15
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    file_path = Column(String, nullable=True)              # pid_model.pt
    status = Column(String, nullable=False, default="active")
    # active / retired / testing / pending_review
    notes = Column(String, nullable=True)
    # --- Wave 1 新增 ---
    kaggle_kernel_url = Column(String, nullable=True)
    epochs = Column(Integer, nullable=True)
    batch_size = Column(Integer, nullable=True)
    arch = Column(String, nullable=True)
    map50_actual = Column(Float, nullable=True)
    map50_95_actual = Column(Float, nullable=True)
    pass_fail = Column(String, nullable=True)              # pass/fail
    accepted_by = Column(String, nullable=True)
    accepted_at = Column(DateTime, nullable=True)
    acceptance_note = Column(String, nullable=True)
    is_current = Column(Boolean, default=False)
    # ---
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
    # Migration: add is_current if not exists (SQLite ALTER TABLE)
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text(
                "ALTER TABLE model_versions ADD COLUMN is_current BOOLEAN DEFAULT FALSE"
            ))
            conn.commit()
        except Exception:
            pass  # 欄位已存在，忽略


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
