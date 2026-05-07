"""Repository for ``ApprovalRequest`` rows."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ApprovalRequest, ApprovalStatus


class ApprovalRepository:
    """All ``ApprovalRequest`` queries live here."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        investigation_id: uuid.UUID,
        prompt_to_reviewer: str,
        proposed_action: dict | None = None,
    ) -> ApprovalRequest:
        approval = ApprovalRequest(
            investigation_id=investigation_id,
            prompt_to_reviewer=prompt_to_reviewer,
            proposed_action=proposed_action,
        )
        self.session.add(approval)
        await self.session.flush()
        return approval

    async def get(self, approval_id: uuid.UUID) -> ApprovalRequest | None:
        return await self.session.get(ApprovalRequest, approval_id)

    async def list_pending(self, *, limit: int = 50) -> Sequence[ApprovalRequest]:
        stmt = (
            select(ApprovalRequest)
            .where(ApprovalRequest.status == ApprovalStatus.PENDING)
            .order_by(ApprovalRequest.created_at.asc())
            .limit(limit)
        )
        return (await self.session.execute(stmt)).scalars().all()

    async def resolve(
        self,
        approval_id: uuid.UUID,
        *,
        approved: bool,
        feedback: str,
    ) -> ApprovalRequest:
        approval = await self.session.get(ApprovalRequest, approval_id)
        if approval is None:
            raise LookupError(f"approval {approval_id} not found")
        approval.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        approval.reviewer_response = feedback
        approval.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
        return approval