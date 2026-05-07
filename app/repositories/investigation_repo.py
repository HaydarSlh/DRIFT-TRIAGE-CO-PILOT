"""Repository for ``Investigation`` and ``InvestigationStep`` rows.

Repositories own all SQLAlchemy queries; services do not import SQLAlchemy directly.
By convention these methods do NOT commit. The service layer owns the unit of work
and decides when to commit; repositories ``flush`` when an id is needed but defer
the commit decision upward.
"""

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Investigation, InvestigationStep, InvestigationStatus


class InvestigationRepository:
    """All ``Investigation`` / ``InvestigationStep`` queries live here."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        drift_event_id: str,
        model_id: str,
        model_version: str = "",
        severity: str = "green",
        thread_id: str,
        created_by: str | None = None,
    ) -> Investigation:
        investigation = Investigation(
            drift_event_id=drift_event_id,
            model_id=model_id,
            model_version=model_version,
            severity=severity,
            thread_id=thread_id,
            created_by=created_by,
        )
        self.session.add(investigation)
        await self.session.flush()
        return investigation

    async def get(self, investigation_id: uuid.UUID) -> Investigation | None:
        stmt = (
            select(Investigation)
            .where(Investigation.id == investigation_id)
            .options(selectinload(Investigation.steps), selectinload(Investigation.approvals))
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_by_thread(self, thread_id: str) -> Investigation | None:
        stmt = select(Investigation).where(Investigation.thread_id == thread_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_by_drift_event_id(self, drift_event_id: str) -> Investigation | None:
        stmt = select(Investigation).where(Investigation.drift_event_id == drift_event_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list(self, *, limit: int = 50) -> Sequence[Investigation]:
        stmt = select(Investigation).order_by(Investigation.created_at.desc()).limit(limit)
        return (await self.session.execute(stmt)).scalars().all()

    async def update_status(self, investigation_id: uuid.UUID, status: InvestigationStatus) -> None:
        investigation = await self.session.get(Investigation, investigation_id)
        if investigation is None:
            raise LookupError(f"investigation {investigation_id} not found")
        investigation.status = status

    async def append_step(
        self,
        *,
        investigation_id: uuid.UUID,
        step_number: int,
        agent_name: str,
        input_summary: dict[str, Any],
        output_summary: dict[str, Any],
        duration_ms: int,
    ) -> InvestigationStep:
        step = InvestigationStep(
            investigation_id=investigation_id,
            step_number=step_number,
            agent_name=agent_name,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=duration_ms,
        )
        self.session.add(step)
        await self.session.flush()
        return step

    async def list_steps(self, investigation_id: uuid.UUID) -> Sequence[InvestigationStep]:
        stmt = (
            select(InvestigationStep)
            .where(InvestigationStep.investigation_id == investigation_id)
            .order_by(InvestigationStep.step_number)
        )
        return (await self.session.execute(stmt)).scalars().all()