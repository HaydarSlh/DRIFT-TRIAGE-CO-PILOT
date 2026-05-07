"""Wire-level Pydantic schemas for the investigations endpoints."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.agents.state import ActionPlan, RetrievedContext
from app.db.models import InvestigationStatus


class InvestigationCreate(BaseModel):
    """Inbound request body for creating an investigation via webhook."""

    drift_event_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = ""
    severity: str = "green"


class InvestigationAccepted(BaseModel):
    """202 response from POST /investigations — the run is queued, not yet finished."""

    id: uuid.UUID
    status: InvestigationStatus
    thread_id: str


class InvestigationStepSummary(BaseModel):
    """One row from the audit trail."""

    model_config = ConfigDict(from_attributes=True)

    step_number: int
    agent_name: str
    duration_ms: int
    created_at: datetime


class InvestigationSummary(BaseModel):
    """Light-weight investigation row used by GET /investigations list responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    drift_event_id: str
    model_id: str
    model_version: str
    severity: str
    status: InvestigationStatus
    thread_id: str
    created_at: datetime
    updated_at: datetime


class InvestigationDetail(InvestigationSummary):
    """Full investigation row — includes the audit trail and final state."""

    steps: list[InvestigationStepSummary] = Field(default_factory=list)
    context: RetrievedContext | None = None
    severity_classified: str | None = None
    recommended_action: ActionPlan | None = None
    comms_draft: str | None = None


class InvestigationResumeRequest(BaseModel):
    """Body for POST /investigations/{id}/resume — primarily a testing affordance."""

    approved: bool = True
    feedback: str = ""