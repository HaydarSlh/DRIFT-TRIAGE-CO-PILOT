"""Triage node — gathers context about a drift event.

The triage agent collects drift context (drift report, recent predictions,
model card, performance metrics) and produces a severity classification
(LOW / MEDIUM / HIGH / CRITICAL) along with a RetrievedContext struct.
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents import llm
from app.agents.prompts.triage import TRIAGE_PROMPT
from app.agents.state import RetrievedContext, TriageState
from app.core.errors import AgentError
from app.core.logging import get_logger

log = get_logger(__name__)


class TriageOutput(BaseModel):
    """Structured triage output."""

    context: RetrievedContext = Field(description="Gathered context about the drift event.")
    severity: str = Field(description="Severity classification: LOW, MEDIUM, HIGH, or CRITICAL.")


async def triage_node(state: TriageState) -> Command[Literal["supervisor"]]:
    """Gather context about the drift event and classify severity.

    Args:
        state: Current graph state. Reads ``drift_event``.

    Returns:
        ``Command(goto="supervisor", update={"context": ..., "severity": ...})``.

    Raises:
        AgentError: If the LLM call fails or returns no structured output.
    """
    drift_event = state.get("drift_event")
    if drift_event is None:
        raise AgentError("triage node called without a drift_event in state")

    model_id = drift_event.model_id
    psi = drift_event.psi
    chi2 = drift_event.chi2
    output_drift = drift_event.output_drift
    severity_hint = drift_event.severity

    chat = llm.get_chat_model("triage").with_structured_output(TriageOutput)
    prompt = TRIAGE_PROMPT.format(
        model_id=model_id,
        psi=psi,
        chi2=chi2,
        output_drift=output_drift,
        severity_hint=severity_hint,
    )

    try:
        output = await chat.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:
        raise AgentError(f"triage failed: {exc}") from exc

    if not isinstance(output, TriageOutput):
        raise AgentError(f"triage returned non-structured output: {type(output).__name__}")

    log.info(
        "triage_done",
        model_id=model_id,
        severity=output.severity,
    )
    return Command(
        goto="supervisor",
        update={"context": output.context, "severity": output.severity},
    )