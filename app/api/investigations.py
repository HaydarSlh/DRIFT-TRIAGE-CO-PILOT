"""Investigation endpoints — async, persisted runs triggered by drift webhooks.

The contract:

- ``POST /webhooks/drift``   receives a DriftAlertEvent, creates an investigation,
                             and kicks off the graph in a background task.
- ``GET /investigations``    lists recent investigations.
- ``GET /investigations/{id}`` returns the full investigation including audit trail
                             and final state (pulled from the checkpointer).
- ``POST /investigations/{id}/resume`` is an internal/testing hook that resumes a
                             paused run without going through the approval API.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.logging import get_logger
from app.deps import get_investigation_service
from app.schemas.investigations import (
    InvestigationAccepted,
    InvestigationDetail,
    InvestigationResumeRequest,
    InvestigationSummary,
)
from app.services.investigation_service import InvestigationService

router = APIRouter(prefix="/investigations", tags=["investigations"])
log = get_logger(__name__)

InvestigationServiceDep = Annotated[InvestigationService, Depends(get_investigation_service)]


@router.get("", response_model=list[InvestigationSummary])
async def list_investigations(service: InvestigationServiceDep) -> list[InvestigationSummary]:
    """List recent investigations, newest first."""
    rows = await service.list_investigations()
    return [InvestigationSummary.model_validate(r) for r in rows]


@router.get("/{investigation_id}", response_model=InvestigationDetail)
async def get_investigation(
    investigation_id: uuid.UUID, service: InvestigationServiceDep
) -> InvestigationDetail:
    """Return a single investigation with its audit trail and final state.

    Final context/severity/action/comms_draft come from the LangGraph
    checkpointer (Redis), not from Postgres — Postgres holds the run record,
    Redis holds the agent state.
    """
    investigation = await service.get_investigation(investigation_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail="investigation not found")
    state_data = await service.get_final_state(investigation.thread_id)
    base = InvestigationSummary.model_validate(investigation)
    return InvestigationDetail(
        **base.model_dump(),
        steps=[s.__dict__ for s in investigation.steps],
        context=state_data.get("context"),
        severity_classified=state_data.get("severity"),
        recommended_action=state_data.get("recommended_action"),
        comms_draft=state_data.get("comms_draft"),
    )


@router.post("/{investigation_id}/resume", response_model=InvestigationSummary)
async def resume_investigation(
    investigation_id: uuid.UUID,
    payload: InvestigationResumeRequest,
    service: InvestigationServiceDep,
) -> InvestigationSummary:
    """Resume a paused run.

    Production resume happens through ``POST /approvals/{id}/respond``; this
    exists so tests (and curious students) can poke the resume path directly.
    """
    investigation = await service.get_investigation(investigation_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail="investigation not found")
    await service.resume_investigation(
        investigation.thread_id,
        {"approved": payload.approved, "feedback": payload.feedback},
    )
    refreshed = await service.get_investigation(investigation_id)
    assert refreshed is not None
    return InvestigationSummary.model_validate(refreshed)