"""add ModelVersion traceability fields: dataset_snapshot_id, train_commit_hash, hyperparams_json

Revision ID: 0004_modelversion_traceability
Revises: 0003_submission_updated_at
Create Date: 2026-04-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0004_modelversion_traceability'
down_revision: Union[str, None] = '0003_submission_updated_at'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('model_versions', sa.Column('dataset_snapshot_id', sa.String(length=255), nullable=True))
    op.add_column('model_versions', sa.Column('train_commit_hash', sa.String(length=40), nullable=True))
    op.add_column('model_versions', sa.Column('hyperparams_json', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_versions', 'hyperparams_json')
    op.drop_column('model_versions', 'train_commit_hash')
    op.drop_column('model_versions', 'dataset_snapshot_id')
