"""External Model Registry — submissions & model_versions 加外部 model 欄位

Revision ID: 0007_external_model_registry
Revises: 0006_m22_discussion
Create Date: 2026-04-26 00:00:00.000000

Changes:
- submissions: add external_source TEXT (e.g. huggingface://meta-llama/Llama-Guard-3-1B)
- model_versions: add external_source TEXT, external_sha256 VARCHAR(64),
                      size_bytes BIGINT, last_used_at TIMESTAMP
- 用途：登記非訓練產出的外部 pretrained model，req_no 用 EXT- prefix 區隔
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0007_external_model_registry'
down_revision: Union[str, None] = '0006_m22_discussion'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # submissions: 記錄外部來源（EXT 工單使用）
    op.add_column(
        'submissions',
        sa.Column('external_source', sa.Text(), nullable=True),
    )

    # model_versions: 外部 model 版本鎖定欄位
    op.add_column(
        'model_versions',
        sa.Column('external_source', sa.Text(), nullable=True),
    )
    op.add_column(
        'model_versions',
        sa.Column('external_sha256', sa.String(64), nullable=True),
    )
    op.add_column(
        'model_versions',
        sa.Column('size_bytes', sa.BigInteger(), nullable=True),
    )
    op.add_column(
        'model_versions',
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('model_versions', 'last_used_at')
    op.drop_column('model_versions', 'size_bytes')
    op.drop_column('model_versions', 'external_sha256')
    op.drop_column('model_versions', 'external_source')
    op.drop_column('submissions', 'external_source')
