"""Shared graph state.

Why a TypedDict (and not a Pydantic BaseModel) for the state itself: LangGraph's
reducers — like ``add_messages`` for the message channel — compose more naturally
with TypedDict. Individual values inside the state can still be Pydantic, and
``RetrievedContext`` and ``ActionPlan`` are, because we want validated structured
output from the triage and action agents.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class DriftEvent(BaseModel):
    """Incoming drift alert event from the platform webhook."""

    event_id: str = Field(description="Unique idempotency key for the webhook event.")
    model_id: str = Field(description="The model that drifted.")
    model_version: str = Field(default="", description="Current production model version.")
    drift_report_id: str = Field(default="", description="Platform drift report reference.")
    psi: dict[str, float] = Field(default_factory=dict, description="PSI per numeric feature.")
    chi2: dict[str, float] = Field(default_factory=dict, description="Chi-squared per categorical feature.")
    output_drift: float = Field(default=0.0, description="Output drift (positive-prediction proportion delta).")
    severity: str = Field(default="green", description="Platform-assessed severity: green/yellow/red.")


class RetrievedContext(BaseModel):
    """Context gathered by the triage agent about the drift event."""

    drift_report_id: str = ""
    recent_predictions_summary: str = ""
    model_card_ref: str = ""
    perf_window_metrics: str = ""


class ActionPlan(BaseModel):
    """Remediation action decided by the action agent."""

    action_type: Literal["none", "replay_test", "retrain_shadow", "rollback"] = Field(
        description="The remediation action to take."
    )
    justification: str = Field(description="Short justification, ≤3 sentences.")
    target_version: str | None = Field(default=None, description="Target version for rollback or retrain.")


class TriageState(TypedDict, total=False):
    """The graph's working memory.

    ``total=False`` because no key is required at every step — only
    ``drift_event`` is set on entry, and the rest are filled in by workers.
    """

    drift_event: DriftEvent
    context: RetrievedContext
    severity: str
    recommended_action: ActionPlan
    comms_draft: str | None
    next_agent: str | None
    step_count: int
    messages: Annotated[list[BaseMessage], add_messages]

    # Persistence / HIL keys:
    # - ``thread_id`` is the LangGraph thread key, also stored on the Investigation
    #   row so we can map a Postgres record to its Redis checkpoint.
    # - ``requires_human_review`` gates the dynamic interrupt inside comms_node.
    #   True only when the action is Production-touching (retrain_shadow, rollback).
    # - ``human_feedback`` is the reviewer's verbatim comment after approve/reject;
    #   the comms agent reads it on resume.
    thread_id: str
    requires_human_review: bool
    human_feedback: str | None