"""M22: Discussion System — submission_comments / submission_attachments / denormalized fields

Revision ID: 0006_m22_discussion
Revises: 0005_m18_cross_resource_retry
Create Date: 2026-04-26 00:00:00.000000

Changes:
- submission_comments table (id, req_no, author_email, body_markdown, is_internal, parent_id, deleted_at, created_at, updated_at)
- submission_attachments table (id, req_no, comment_id, filename, size_bytes, mime_type, storage_path, uploaded_by, uploaded_at)
- submissions: add discussion_count (INT default 0), last_activity_at (TIMESTAMP nullable)
- index: (req_no, created_at) on submission_comments
- index: (req_no, created_at) on submission_attachments
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0006_m22_discussion'
down_revision: Union[str, None] = '0005_m18_cross_resource_retry'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # submission_comments table
    op.create_table(
        'submission_comments',
        sa.Column('id', sa.Integer(), primary_key=True, index=True, nullable=False),
        sa.Column('req_no', sa.String(), nullable=False),
        sa.Column('author_email', sa.String(), nullable=False),
        sa.Column('body_markdown', sa.Text(), nullable=False),
        sa.Column('is_internal', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('parent_id', sa.Integer(), sa.ForeignKey('submission_comments.id'), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index(
        'ix_submission_comments_req_no_created_at',
        'submission_comments',
        ['req_no', 'created_at'],
    )

    # submission_attachments table
    op.create_table(
        'submission_attachments',
        sa.Column('id', sa.Integer(), primary_key=True, index=True, nullable=False),
        sa.Column('req_no', sa.String(), nullable=False),
        sa.Column('comment_id', sa.Integer(), sa.ForeignKey('submission_comments.id'), nullable=True),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('mime_type', sa.String(), nullable=False),
        sa.Column('storage_path', sa.String(), nullable=False),
        sa.Column('uploaded_by', sa.String(), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(), nullable=True),
    )
    op.create_index(
        'ix_submission_attachments_req_no_uploaded_at',
        'submission_attachments',
        ['req_no', 'uploaded_at'],
    )

    # submissions: denormalized discussion fields
    op.add_column(
        'submissions',
        sa.Column('discussion_count', sa.Integer(), nullable=False, server_default='0'),
    )
    op.add_column(
        'submissions',
        sa.Column('last_activity_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('submissions', 'last_activity_at')
    op.drop_column('submissions', 'discussion_count')
    op.drop_index('ix_submission_attachments_req_no_uploaded_at', table_name='submission_attachments')
    op.drop_table('submission_attachments')
    op.drop_index('ix_submission_comments_req_no_created_at', table_name='submission_comments')
    op.drop_table('submission_comments')
