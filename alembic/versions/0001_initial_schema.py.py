"""Initial CP2 schema — reports, report_runs, review_requests.

Hand-written from a reviewed autogenerate output. Edits applied:
- ``server_default=sa.func.now()`` on every ``created_at`` so the database
  stamps the timestamp; otherwise rows inserted via raw SQL get NULL.
- Explicit composite/standalone indexes the autogenerator does not emit:
  ``ix_reports_status``, ``ix_report_runs_report_id``,
  ``ix_review_requests_report_id``, ``ix_review_requests_status``.
- Postgres enum types named explicitly so they're nameable in psql and
  reusable across migrations.

Revision ID: 0001
Revises:
Create Date: 2026-05-01 12:00:00
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
    report_status = postgresql.ENUM(
        "pending",
        "running",
        "awaiting_review",
        "completed",
        "failed",
        name="report_status",
        create_type=False,
    )
    review_status = postgresql.ENUM(
        "pending",
        "approved",
        "rejected",
        name="review_status",
        create_type=False,
    )
    report_status.create(op.get_bind(), checkfirst=True)
    review_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("topic", sa.String(length=500), nullable=False),
        sa.Column("status", report_status, nullable=False, server_default="pending"),
        sa.Column("thread_id", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("thread_id", name="uq_reports_thread_id"),
    )
    op.create_index("ix_reports_status", "reports", ["status"])

    op.create_table(
        "report_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step_number", sa.Integer(), nullable=False),
        sa.Column("agent_name", sa.String(length=64), nullable=False),
        sa.Column(
            "input_summary",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "output_summary",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_report_runs_report_id", "report_runs", ["report_id"])

    op.create_table(
        "review_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("prompt_to_reviewer", sa.Text(), nullable=False),
        sa.Column("reviewer_response", sa.Text(), nullable=True),
        sa.Column("status", review_status, nullable=False, server_default="pending"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=False), nullable=True),
    )
    op.create_index("ix_review_requests_report_id", "review_requests", ["report_id"])
    op.create_index("ix_review_requests_status", "review_requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_review_requests_status", table_name="review_requests")
    op.drop_index("ix_review_requests_report_id", table_name="review_requests")
    op.drop_table("review_requests")

    op.drop_index("ix_report_runs_report_id", table_name="report_runs")
    op.drop_table("report_runs")

    op.drop_index("ix_reports_status", table_name="reports")
    op.drop_table("reports")

    op.execute("DROP TYPE IF EXISTS review_status")
    op.execute("DROP TYPE IF EXISTS report_status")
