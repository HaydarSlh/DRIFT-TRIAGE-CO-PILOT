"""Shared Pydantic schemas for the agent API layer."""
from pydantic import BaseModel


class AgentState(BaseModel):
    model_id: str
    drift_score: float
    triage_decision: str | None = None
    action_taken: str | None = None
    comms_sent: bool = False
