"""M16: add epoch_curve column to submissions for training curve visualization

Revision ID: 0005_m16_epoch_curve
Revises: 0004_modelversion_traceability
Create Date: 2026-04-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0005_m16_epoch_curve'
down_revision: Union[str, None] = '0004_modelversion_traceability'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # TEXT JSON: [{"epoch":1,"train_loss":0.8,"val_loss":0.5,"map50":0.3,"map50_95":0.15}, ...]
    op.add_column('submissions', sa.Column('epoch_curve', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('submissions', 'epoch_curve')
