"""Approval service — receives reviewer responses and resumes the graph.

Approving an approval request feeds ``{"approved": True, "feedback": ...}`` into the
comms dynamic interrupt; rejecting feeds ``{"approved": False, ...}``. The comms
node distinguishes the two and either dispatches the action or ends without it.
"""

import uuid
from collections.abc import Sequence
from typing import Any

from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.db.models import ApprovalRequest, ApprovalStatus
from app.repositories.approval_repo import ApprovalRepository
from app.repositories.investigation_repo import InvestigationRepository
from app.services.investigation_service import InvestigationService

log = get_logger(__name__)


class ApprovalService:
    """High-level operations on approval requests."""

    def __init__(
        self,
        sessionmaker: async_sessionmaker,
        investigation_service: InvestigationService,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._investigation_service = investigation_service

    async def list_pending(self) -> Sequence[ApprovalRequest]:
        async with self._sessionmaker() as session:
            return await ApprovalRepository(session).list_pending()

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        async with self._sessionmaker() as session:
            return await ApprovalRepository(session).get(approval_id)

    async def submit_response(
        self,
        approval_id: uuid.UUID,
        *,
        approved: bool,
        feedback: str,
    ) -> ApprovalRequest:
        """Resolve an approval request and resume the corresponding graph run."""
        async with self._sessionmaker() as session:
            approval_repo = ApprovalRepository(session)
            investigation_repo = InvestigationRepository(session)
            approval = await approval_repo.get(approval_id)
            if approval is None:
                raise ValidationError(f"approval {approval_id} not found")
            if approval.status is not ApprovalStatus.PENDING:
                raise ValidationError(f"approval {approval_id} is already {approval.status}")

            investigation = await investigation_repo.get(approval.investigation_id)
            if investigation is None:
                raise ValidationError(f"approval {approval_id} has no parent investigation")
            thread_id = investigation.thread_id

            await approval_repo.resolve(approval_id, approved=approved, feedback=feedback)
            await session.commit()
            await session.refresh(approval)

        log.info(
            "approval_resolved",
            approval_id=str(approval_id),
            approved=approved,
            thread_id=thread_id,
        )

        await self._investigation_service.resume_investigation(
            thread_id, {"approved": approved, "feedback": feedback}
        )
        return approval

    async def submit_response_async(
        self,
        approval_id: uuid.UUID,
        *,
        approved: bool,
        feedback: str,
        background_tasks: BackgroundTasks,
    ) -> ApprovalRequest:
        """Resolve the DB row immediately; schedule graph resume as a background task.

        The HTTP response returns as soon as the row is committed — the LLM calls
        and tool dispatch happen after the response is sent.
        """
        async with self._sessionmaker() as session:
            approval_repo = ApprovalRepository(session)
            investigation_repo = InvestigationRepository(session)
            approval = await approval_repo.get(approval_id)
            if approval is None:
                raise ValidationError(f"approval {approval_id} not found")
            if approval.status is not ApprovalStatus.PENDING:
                raise ValidationError(f"approval {approval_id} is already {approval.status}")

            investigation = await investigation_repo.get(approval.investigation_id)
            if investigation is None:
                raise ValidationError(f"approval {approval_id} has no parent investigation")
            thread_id = investigation.thread_id

            await approval_repo.resolve(approval_id, approved=approved, feedback=feedback)
            await session.commit()
            await session.refresh(approval)

        log.info(
            "approval_resolved_async",
            approval_id=str(approval_id),
            approved=approved,
            thread_id=thread_id,
        )

        decision: dict[str, Any] = {"approved": approved, "feedback": feedback}
        background_tasks.add_task(
            self._investigation_service.resume_investigation, thread_id, decision
        )
        return approval