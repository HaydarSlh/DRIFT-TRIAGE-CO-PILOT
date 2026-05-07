"""Wire-level Pydantic schemas for the approval endpoints."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import ApprovalStatus


class ApprovalSummary(BaseModel):
    """One pending or resolved approval request."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    investigation_id: uuid.UUID
    prompt_to_reviewer: str
    proposed_action: dict | None
    reviewer_response: str | None
    status: ApprovalStatus
    created_at: datetime
    resolved_at: datetime | None


class ApprovalResponseRequest(BaseModel):
    """Body for POST /approvals/{id}/respond."""

    approved: bool
    feedback: str = Field(default="", max_length=2000)