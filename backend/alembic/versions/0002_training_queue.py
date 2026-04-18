"""training_queue table

Revision ID: 0002_training_queue
Revises: 0001_initial
Create Date: 2026-04-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0002_training_queue'
down_revision: Union[str, None] = '0001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'training_queue',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('req_no', sa.String(), nullable=False),
        sa.Column('priority', sa.String(), nullable=False, server_default='P2'),
        sa.Column('status', sa.String(), nullable=False, server_default='waiting'),
        sa.Column('enqueued_at', sa.DateTime(), nullable=False),
        sa.Column('dispatched_at', sa.DateTime(), nullable=True),
        sa.Column('target_resource', sa.String(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('error_reason', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('req_no'),
    )
    op.create_index(op.f('ix_training_queue_id'), 'training_queue', ['id'], unique=False)
    op.create_index(op.f('ix_training_queue_req_no'), 'training_queue', ['req_no'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_training_queue_req_no'), table_name='training_queue')
    op.drop_index(op.f('ix_training_queue_id'), table_name='training_queue')
    op.drop_table('training_queue')
