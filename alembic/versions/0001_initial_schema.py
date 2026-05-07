"""Initial drift triage schema — investigations, investigation_steps, approval_requests.

Revision ID: 0001
Revises:
Create Date: 2026-05-07
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    investigation_status = postgresql.ENUM(
        "pending", "running", "awaiting_approval", "completed", "failed",
        name="investigation_status",
        create_type=False,
    )
    approval_status = postgresql.ENUM(
        "pending", "approved", "rejected",
        name="approval_status",
        create_type=False,
    )
    investigation_status.create(op.get_bind(), checkfirst=True)
    approval_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "investigations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("drift_event_id", sa.String(255), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False, server_default=""),
        sa.Column("severity", sa.String(32), nullable=False, server_default="green"),
        sa.Column("status", investigation_status, nullable=False, server_default="pending"),
        sa.Column("thread_id", sa.String(64), nullable=False),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("drift_event_id", name="uq_investigations_drift_event_id"),
        sa.UniqueConstraint("thread_id", name="uq_investigations_thread_id"),
    )
    op.create_index("ix_investigations_status", "investigations", ["status"])

    op.create_table(
        "investigation_steps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step_number", sa.Integer(), nullable=False),
        sa.Column("agent_name", sa.String(64), nullable=False),
        sa.Column("input_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("output_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_investigation_steps_investigation_id", "investigation_steps", ["investigation_id"])

    op.create_table(
        "approval_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("prompt_to_reviewer", sa.Text(), nullable=False),
        sa.Column("proposed_action", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("reviewer_response", sa.Text(), nullable=True),
        sa.Column("status", approval_status, nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=False), nullable=True),
    )
    op.create_index("ix_approval_requests_investigation_id", "approval_requests", ["investigation_id"])
    op.create_index("ix_approval_requests_status", "approval_requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_approval_requests_status", table_name="approval_requests")
    op.drop_index("ix_approval_requests_investigation_id", table_name="approval_requests")
    op.drop_table("approval_requests")

    op.drop_index("ix_investigation_steps_investigation_id", table_name="investigation_steps")
    op.drop_table("investigation_steps")

    op.drop_index("ix_investigations_status", table_name="investigations")
    op.drop_table("investigations")

    op.execute("DROP TYPE IF EXISTS approval_status")
    op.execute("DROP TYPE IF EXISTS investigation_status")