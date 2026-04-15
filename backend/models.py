from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
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
    product = Column(String, nullable=False)       # AICAD / 天機 / RS 等
    company = Column(String, nullable=False)       # 業務方公司，如 HurricaneEdge
    submitter = Column(String, nullable=True)      # 業務方聯絡人
    purpose = Column(String, nullable=True)        # 業務問題描述
    class_list = Column(String, nullable=True)     # 類別清單（逗號分隔）
    map50_threshold = Column(Float, nullable=True)  # 驗收門檻 mAP50
    input_spec = Column(String, nullable=True)     # 輸入規格描述
    deploy_env = Column(String, nullable=True)     # 部署目標，如 aicad-api :8200
    dataset_source = Column(String, nullable=True)  # 訓練資料來源
    dataset_count = Column(String, nullable=True)  # 資料集大小描述
    label_format = Column(String, nullable=True)   # 標注格式，如 YOLO
    expected_delivery = Column(String, nullable=True)  # 預計交付日期
    status = Column(String, nullable=False, default="pending")
    # pending / in_progress / completed / cancelled
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    """模型版本清冊"""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    req_no = Column(String, index=True, nullable=False)   # 對應需求單
    product = Column(String, nullable=False)
    model_name = Column(String, nullable=False)           # PID 管線辨識
    version = Column(String, nullable=False)              # v2
    train_date = Column(String, nullable=True)            # 2026-04-15
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    file_path = Column(String, nullable=True)             # pid_model.pt
    status = Column(String, nullable=False, default="active")
    # active / retired / testing / pending_review
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
