"""
seed_data.py — ModelHub Phase 0 初始資料種子

資料來源：~/HurricaneGroup/docs/hurricanecore/modelhub/
  - model-registry.md
  - submissions/MH-2026-001-aicad-pid.md
  - submissions/MH-2026-002-aicad-instrument.md

執行方式：
  python seed_data.py
"""

import sys
import os

from models import Submission, ModelVersion, init_db, SessionLocal

SUBMISSIONS = [
    {
        "req_no": "MH-2026-001",
        "req_name": "P&ID 管線辨識",
        "product": "AICAD",
        "company": "HurricaneEdge",
        "submitter": None,                          # 待 HurricaneEdge 補填
        "purpose": "P&ID 工程圖管線辨識，AICAD 轉檔流程核心辨識步驟",
        "priority": "P1",
        "model_type": "detection",
        "class_list": None,                         # 待 HurricaneEdge 補填
        "map50_threshold": None,
        "map50_target": None,                       # 待 HurricaneEdge 補填
        "arch": "yolov8m",
        "input_spec": "P&ID 工程圖 PNG 格式（或 PDF 轉 PNG），解析度範圍待補填",
        "deploy_env": "aicad-api :8200",
        "dataset_source": None,                     # 待 HurricaneEdge 補填
        "dataset_count": None,                      # 待 HurricaneEdge 補填
        "label_format": "YOLO（推測，待確認）",
        "expected_delivery": "已部署（Phase 0 補登記，部署日期 2026-04-15）",
        "status": "accepted",
    },
    {
        "req_no": "MH-2026-002",
        "req_name": "P&ID 儀器符號辨識",
        "product": "AICAD",
        "company": "HurricaneEdge",
        "submitter": None,
        "purpose": "P&ID 工程圖儀器符號辨識，與 PID 管線辨識模型協同運作",
        "priority": "P1",
        "model_type": "detection",
        "class_list": None,                         # 待 HurricaneEdge 補填（儀器符號類型清單）
        "map50_threshold": None,
        "map50_target": None,
        "arch": "yolov8m",
        "input_spec": "P&ID 工程圖 PNG 格式（或 PDF 轉 PNG），解析度範圍待補填",
        "deploy_env": "aicad-api :8200",
        "dataset_source": None,
        "dataset_count": None,
        "label_format": "YOLO（推測，待確認）",
        "expected_delivery": "已部署（Phase 0 補登記，部署日期 2026-04-15）",
        "status": "accepted",
    },
]

MODEL_VERSIONS = [
    {
        "req_no": "MH-2026-001",
        "product": "AICAD",
        "model_name": "PID 管線辨識",
        "version": "v2",
        "train_date": "2026-04-15",
        "map50": 0.989,
        "map50_95": 0.967,
        "map50_actual": 0.989,
        "map50_95_actual": 0.967,
        "pass_fail": "pass",
        "arch": "yolov8m",
        "file_path": "pid_model.pt",
        "status": "active",
        "notes": "Phase 0 補登記，當前生產版本。mAP50-95=0.967。"
                 "YOLOv8 架構（具體型號待 HurricaneEdge 確認）。"
                 "驗收標準尚待 HurricaneEdge 正式簽核。",
    },
    {
        "req_no": "MH-2026-002",
        "product": "AICAD",
        "model_name": "儀器辨識",
        "version": "v2",
        "train_date": "2026-04-15",
        "map50": 0.895,
        "map50_95": None,       # 資料缺失，待 HurricaneEdge 補充
        "map50_actual": 0.895,
        "map50_95_actual": None,
        "pass_fail": "pass",
        "arch": "yolov8m",
        "file_path": "instrument_yolo.pt",
        "status": "active",
        "notes": "Phase 0 補登記，當前生產版本。mAP50-95 資料缺失，待 HurricaneEdge 補充或說明。"
                 "YOLOv8 架構（具體型號待確認）。"
                 "驗收標準尚待 HurricaneEdge 正式簽核。",
    },
]


def seed():
    init_db()
    db = SessionLocal()
    inserted = {"submissions": 0, "model_versions": 0}

    for data in SUBMISSIONS:
        existing = db.query(Submission).filter(Submission.req_no == data["req_no"]).first()
        if existing:
            print(f"  SKIP Submission {data['req_no']} (already exists)")
            continue
        db.add(Submission(**data))
        inserted["submissions"] += 1
        print(f"  INSERT Submission {data['req_no']}")

    for data in MODEL_VERSIONS:
        existing = (
            db.query(ModelVersion)
            .filter(
                ModelVersion.req_no == data["req_no"],
                ModelVersion.version == data["version"],
            )
            .first()
        )
        if existing:
            print(f"  SKIP ModelVersion {data['req_no']} {data['version']} (already exists)")
            continue
        db.add(ModelVersion(**data))
        inserted["model_versions"] += 1
        print(f"  INSERT ModelVersion {data['req_no']} {data['version']}")

    db.commit()
    db.close()
    print(
        f"\nSeed complete: {inserted['submissions']} submissions, "
        f"{inserted['model_versions']} model_versions inserted."
    )


if __name__ == "__main__":
    print("ModelHub Phase 0 seed starting...")
    seed()
