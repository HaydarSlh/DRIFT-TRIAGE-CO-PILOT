"""ORM models for the drift triage persistence layer.

Three tables:

- ``investigations``        — one row per drift webhook; the durable record of
                             "what did the platform alert us about, and how did we triage it?"
- ``investigation_steps``   — append-only audit trail of agent invocations within
                             an investigation.
- ``approval_requests``     — outstanding human-in-the-loop approval requests
                             (only created when the action is Production-touching).

The graph's *short-term* memory (per-step state, cursor, intermediate tool calls)
lives in Redis under the LangGraph checkpointer, keyed by ``Investigation.thread_id``.
Postgres holds the *long-term* domain record.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class InvestigationStatus(enum.StrEnum):
    """Lifecycle states of an investigation."""

    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalStatus(enum.StrEnum):
    """Lifecycle states of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Investigation(Base):
    """A drift-event-triggered investigation.

    ``thread_id`` links this row to the LangGraph checkpoint in Redis.
    ``drift_event_id`` is the idempotency key from the platform's webhook,
    so duplicate redeliveries are no-ops.
    """

    __tablename__ = "investigations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    drift_event_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True,
    )
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="green")
    status: Mapped[InvestigationStatus] = mapped_column(
        SAEnum(
            InvestigationStatus,
            name="investigation_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=InvestigationStatus.PENDING,
    )
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    steps: Mapped[list["InvestigationStep"]] = relationship(
        back_populates="investigation",
        cascade="all, delete-orphan",
        order_by="InvestigationStep.step_number",
    )
    approvals: Mapped[list["ApprovalRequest"]] = relationship(
        back_populates="investigation",
        cascade="all, delete-orphan",
        order_by="ApprovalRequest.created_at",
    )

    __table_args__ = (
        Index("ix_investigations_status", "status"),
    )


class InvestigationStep(Base):
    """One recorded agent invocation within an investigation's run.

    Append-only — these rows are never updated, only inserted.
    """

    __tablename__ = "investigation_steps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False,
    )
    step_number: Mapped[int] = mapped_column(nullable=False)
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    input_summary: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    output_summary: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    duration_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )

    investigation: Mapped["Investigation"] = relationship(back_populates="steps")

    __table_args__ = (
        Index("ix_investigation_steps_investigation_id", "investigation_id"),
    )


class ApprovalRequest(Base):
    """A pending or resolved human approval request.

    Created when the comms node hits an interrupt that requires a human.
    Resolved when a reviewer responds via ``POST /approvals/{id}/respond``.
    The ``proposed_action`` JSONB column stores the action plan so the
    dashboard can render the HIL prompt without re-reading the checkpoint.
    """

    __tablename__ = "approval_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False,
    )
    prompt_to_reviewer: Mapped[str] = mapped_column(Text, nullable=False)
    proposed_action: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    reviewer_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ApprovalStatus] = mapped_column(
        SAEnum(
            ApprovalStatus,
            name="approval_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=ApprovalStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(nullable=True)

    investigation: Mapped["Investigation"] = relationship(back_populates="approvals")

    __table_args__ = (
        Index("ix_approval_requests_investigation_id", "investigation_id"),
        Index("ix_approval_requests_status", "status"),
    )