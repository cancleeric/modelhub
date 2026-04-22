"""add submissions.updated_at column

Revision ID: 0003_submission_updated_at
Revises: 0002_training_queue
Create Date: 2026-04-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0003_submission_updated_at'
down_revision: Union[str, None] = '0002_training_queue'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('submissions', sa.Column('updated_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('submissions', 'updated_at')
