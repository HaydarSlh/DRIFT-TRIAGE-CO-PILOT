"""Supervisor node — the only router in the graph."""

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents import llm
from app.agents.prompts.supervisor import SUPERVISOR_PROMPT
from app.agents.state import TriageState
from app.core.errors import AgentError
from app.core.logging import get_logger

log = get_logger(__name__)

MAX_STEPS = 8


class SupervisorDecision(BaseModel):
    """Structured supervisor output.

    Restricting ``next_agent`` to a Literal means we never have to parse free-form
    text or guess intent — invalid values are rejected by Pydantic before the node
    can act on them.
    """

    next_agent: Literal["triage", "action", "comms", "done"] = Field(
        description="Which worker should run next, or 'done' to end the run."
    )
    reasoning: str = Field(description="One short sentence explaining the choice.")


async def supervisor_node(
    state: TriageState,
) -> Command[Literal["triage", "action", "comms", "__end__"]]:
    """Decide the next worker to run, or end the graph.

    Increments ``step_count`` on every call. If ``step_count`` exceeds ``MAX_STEPS``
    after incrementing, force END regardless of the LLM's decision — this is the
    runaway-loop guardrail.

    Args:
        state: Current graph state.

    Returns:
        ``Command`` with ``goto`` set to the next node or ``END``, and an update that
        bumps ``step_count`` and records the decision.

    Raises:
        AgentError: If the supervisor LLM call fails.
    """
    step_count = state.get("step_count", 0) + 1
    if step_count > MAX_STEPS:
        log.warning("supervisor_max_steps_reached", step_count=step_count)
        return Command(
            goto=END,
            update={"step_count": step_count, "next_agent": "done"},
        )

    chat = llm.get_chat_model("supervisor").with_structured_output(SupervisorDecision)
    drift_event = state.get("drift_event")
    model_id = drift_event.model_id if drift_event else ""
    severity = state.get("severity", "")
    action = state.get("recommended_action")
    action_type = action.action_type if action else ""
    prompt = SUPERVISOR_PROMPT.format(
        model_id=model_id,
        severity=severity,
        action_type=action_type,
        has_context=bool(state.get("context")),
        step_count=step_count,
    )
    try:
        decision = await chat.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:
        raise AgentError(f"supervisor decision failed: {exc}") from exc

    if not isinstance(decision, SupervisorDecision):
        raise AgentError(f"supervisor returned non-decision: {type(decision).__name__}")

    log.info(
        "supervisor_decision",
        next_agent=decision.next_agent,
        reasoning=decision.reasoning,
        step_count=step_count,
    )
    goto: str = END if decision.next_agent == "done" else decision.next_agent
    return Command(
        goto=goto,
        update={"step_count": step_count, "next_agent": decision.next_agent},
    )