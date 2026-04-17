"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # submissions table
    op.create_table(
        'submissions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('req_no', sa.String(), nullable=False),
        sa.Column('req_name', sa.String(), nullable=True),
        sa.Column('product', sa.String(), nullable=False),
        sa.Column('company', sa.String(), nullable=False),
        sa.Column('submitter', sa.String(), nullable=True),
        sa.Column('purpose', sa.String(), nullable=True),
        sa.Column('priority', sa.String(), nullable=False),
        sa.Column('model_type', sa.String(), nullable=True),
        sa.Column('class_list', sa.String(), nullable=True),
        sa.Column('map50_threshold', sa.Float(), nullable=True),
        sa.Column('map50_target', sa.Float(), nullable=True),
        sa.Column('map50_95_target', sa.Float(), nullable=True),
        sa.Column('inference_latency_ms', sa.Integer(), nullable=True),
        sa.Column('model_size_limit_mb', sa.Integer(), nullable=True),
        sa.Column('arch', sa.String(), nullable=True),
        sa.Column('input_spec', sa.String(), nullable=True),
        sa.Column('deploy_env', sa.String(), nullable=True),
        sa.Column('dataset_source', sa.String(), nullable=True),
        sa.Column('dataset_count', sa.String(), nullable=True),
        sa.Column('dataset_val_count', sa.Integer(), nullable=True),
        sa.Column('dataset_test_count', sa.Integer(), nullable=True),
        sa.Column('class_count', sa.Integer(), nullable=True),
        sa.Column('label_format', sa.String(), nullable=True),
        sa.Column('kaggle_dataset_url', sa.String(), nullable=True),
        sa.Column('dataset_path', sa.String(), nullable=True),
        sa.Column('dataset_train_count', sa.Integer(), nullable=True),
        sa.Column('model_output_path', sa.String(), nullable=True),
        sa.Column('expected_delivery', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('reviewer_note', sa.String(), nullable=True),
        sa.Column('reviewed_by', sa.String(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(), nullable=True),
        sa.Column('rejection_reasons', sa.String(), nullable=True),
        sa.Column('rejection_note', sa.String(), nullable=True),
        sa.Column('resubmit_count', sa.Integer(), nullable=True),
        sa.Column('resubmitted_at', sa.DateTime(), nullable=True),
        sa.Column('kaggle_kernel_slug', sa.String(), nullable=True),
        sa.Column('kaggle_kernel_version', sa.Integer(), nullable=True),
        sa.Column('kaggle_status', sa.String(), nullable=True),
        sa.Column('kaggle_status_updated_at', sa.DateTime(), nullable=True),
        sa.Column('kaggle_log_url', sa.String(), nullable=True),
        sa.Column('training_started_at', sa.DateTime(), nullable=True),
        sa.Column('training_completed_at', sa.DateTime(), nullable=True),
        sa.Column('gpu_seconds', sa.Integer(), nullable=True),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=True),
        sa.Column('total_attempts', sa.Integer(), nullable=True),
        sa.Column('max_retries', sa.Integer(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=True),
        sa.Column('max_budget_usd', sa.Float(), nullable=True),
        sa.Column('budget_exceeded_notified', sa.Boolean(), nullable=True),
        sa.Column('dataset_status', sa.String(), nullable=False),
        sa.Column('blocked_reason', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_submissions_id'), 'submissions', ['id'], unique=False)
    op.create_index(op.f('ix_submissions_req_no'), 'submissions', ['req_no'], unique=True)

    # model_versions table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('req_no', sa.String(), nullable=False),
        sa.Column('product', sa.String(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('train_date', sa.String(), nullable=True),
        sa.Column('map50', sa.Float(), nullable=True),
        sa.Column('map50_95', sa.Float(), nullable=True),
        sa.Column('file_path', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('notes', sa.String(), nullable=True),
        sa.Column('kaggle_kernel_url', sa.String(), nullable=True),
        sa.Column('epochs', sa.Integer(), nullable=True),
        sa.Column('batch_size', sa.Integer(), nullable=True),
        sa.Column('arch', sa.String(), nullable=True),
        sa.Column('map50_actual', sa.Float(), nullable=True),
        sa.Column('map50_95_actual', sa.Float(), nullable=True),
        sa.Column('pass_fail', sa.String(), nullable=True),
        sa.Column('accepted_by', sa.String(), nullable=True),
        sa.Column('accepted_at', sa.DateTime(), nullable=True),
        sa.Column('acceptance_note', sa.String(), nullable=True),
        sa.Column('is_current', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_model_versions_id'), 'model_versions', ['id'], unique=False)
    op.create_index(op.f('ix_model_versions_req_no'), 'model_versions', ['req_no'], unique=False)

    # api_keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('created_by', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('disabled', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_api_keys_id'), 'api_keys', ['id'], unique=False)
    op.create_index(op.f('ix_api_keys_key'), 'api_keys', ['key'], unique=True)

    # submission_history table
    op.create_table(
        'submission_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('req_no', sa.String(), nullable=False),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('actor', sa.String(), nullable=True),
        sa.Column('reasons', sa.String(), nullable=True),
        sa.Column('note', sa.String(), nullable=True),
        sa.Column('meta', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_submission_history_created_at'), 'submission_history', ['created_at'], unique=False)
    op.create_index(op.f('ix_submission_history_id'), 'submission_history', ['id'], unique=False)
    op.create_index(op.f('ix_submission_history_req_no'), 'submission_history', ['req_no'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_submission_history_req_no'), table_name='submission_history')
    op.drop_index(op.f('ix_submission_history_id'), table_name='submission_history')
    op.drop_index(op.f('ix_submission_history_created_at'), table_name='submission_history')
    op.drop_table('submission_history')
    op.drop_index(op.f('ix_api_keys_key'), table_name='api_keys')
    op.drop_index(op.f('ix_api_keys_id'), table_name='api_keys')
    op.drop_table('api_keys')
    op.drop_index(op.f('ix_model_versions_req_no'), table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_id'), table_name='model_versions')
    op.drop_table('model_versions')
    op.drop_index(op.f('ix_submissions_req_no'), table_name='submissions')
    op.drop_index(op.f('ix_submissions_id'), table_name='submissions')
    op.drop_table('submissions')
