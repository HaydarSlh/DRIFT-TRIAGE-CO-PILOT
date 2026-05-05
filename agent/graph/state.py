from typing import TypedDict


class TriageState(TypedDict):
    model_id: str
    drift_score: float
    triage_decision: str | None
    action_taken: str | None
    comms_sent: bool
