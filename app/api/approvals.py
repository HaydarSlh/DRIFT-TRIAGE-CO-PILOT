"""Human-approval endpoints.

The graph pauses on a dynamic interrupt inside ``comms_node`` when the action
is Production-touching (retrain_shadow or rollback). Each pause creates an
``ApprovalRequest`` row in ``pending`` status. A reviewer responds via
``POST /approvals/{id}/respond``; that resolves the row AND resumes the graph
with the reviewer's verdict fed back into the interrupt.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.logging import get_logger
from app.deps import get_approval_service
from app.schemas.approvals import ApprovalResponseRequest, ApprovalSummary
from app.services.approval_service import ApprovalService

router = APIRouter(prefix="/approvals", tags=["approvals"])
log = get_logger(__name__)

ApprovalServiceDep = Annotated[ApprovalService, Depends(get_approval_service)]


@router.get("/pending", response_model=list[ApprovalSummary])
async def list_pending(service: ApprovalServiceDep) -> list[ApprovalSummary]:
    """Return unresolved approval requests, oldest first (queue semantics)."""
    rows = await service.list_pending()
    return [ApprovalSummary.model_validate(r) for r in rows]


@router.post("/{approval_id}/respond", response_model=ApprovalSummary)
async def respond(
    approval_id: uuid.UUID,
    payload: ApprovalResponseRequest,
    service: ApprovalServiceDep,
) -> ApprovalSummary:
    """Resolve an approval and resume its graph run with the verdict."""
    approval = await service.get(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="approval not found")
    updated = await service.submit_response(
        approval_id, approved=payload.approved, feedback=payload.feedback
    )
    return ApprovalSummary.model_validate(updated)