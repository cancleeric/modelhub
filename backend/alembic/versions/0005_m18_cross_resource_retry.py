"""M18: Cross-Resource Auto Retry — training_queue.attempted_resources + events table

Revision ID: 0005_m18_cross_resource_retry
Revises: 0004_modelversion_traceability
Create Date: 2026-04-26 00:00:00.000000

Changes:
- training_queue: add attempted_resources (TEXT, default '[]')
- create events table (SystemEvent) for brain-console UI
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0005_m18_cross_resource_retry'
down_revision: Union[str, None] = '0004_modelversion_traceability'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # M18-1: training_queue.attempted_resources
    op.add_column(
        'training_queue',
        sa.Column('attempted_resources', sa.Text(), nullable=True, server_default='[]'),
    )

    # M18-4: events table
    op.create_table(
        'events',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('event_type', sa.String(), nullable=False, index=True),
        sa.Column('req_no', sa.String(), nullable=True, index=True),
        sa.Column('severity', sa.String(), nullable=False, server_default='warning'),
        sa.Column('message', sa.String(), nullable=False),
        sa.Column('meta', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('events')
    op.drop_column('training_queue', 'attempted_resources')
