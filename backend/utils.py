"""
utils.py — ModelHub 共用工具函式

P1-8: 抽出各 poller 共用的 _next_version_for()，避免三份重複實作。
"""

import re
from sqlalchemy.orm import Session


def next_version_for(db: Session, req_no: str) -> str:
    """
    查詢 req_no 目前最大 ModelVersion，回傳下一個版號字串（v1, v2, ...）。

    Args:
        db:     SQLAlchemy Session
        req_no: 需求單編號

    Returns:
        "v1" 若無任何既有版本，否則 "v{n+1}"
    """
    from models import ModelVersion
    latest = (
        db.query(ModelVersion)
        .filter(ModelVersion.req_no == req_no)
        .order_by(ModelVersion.id.desc())
        .first()
    )
    if not latest:
        return "v1"
    m = re.match(r"v(\d+)", latest.version or "")
    n = int(m.group(1)) + 1 if m else 1
    return f"v{n}"
