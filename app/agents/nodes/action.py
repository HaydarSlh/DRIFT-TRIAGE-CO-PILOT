"""Action node — decides the remediation action.

Takes the drift event and gathered context, then proposes an ActionPlan.
Only Production-touching actions (retrain_shadow, rollback) set
``requires_human_review`` to True, which triggers the HIL interrupt
in the comms node downstream.
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agents import llm
from app.agents.prompts.action import ACTION_PROMPT
from app.agents.state import ActionPlan, TriageState
from app.core.errors import AgentError
from app.core.logging import get_logger

log = get_logger(__name__)

PRODUCTION_TOUCHING_ACTIONS = {"retrain_shadow", "rollback"}


async def action_node(state: TriageState) -> Command[Literal["supervisor"]]:
    """Propose a remediation action based on the drift event and context.

    Args:
        state: Current graph state. Reads ``drift_event``, ``context``, ``severity``.

    Returns:
        ``Command(goto="supervisor", update={"recommended_action": ..., "requires_human_review": ...})``.

    Raises:
        AgentError: If the LLM call fails.
    """
    drift_event = state.get("drift_event")
    context = state.get("context")
    severity = state.get("severity", "LOW")

    model_id = drift_event.model_id if drift_event else ""
    drift_details = ""
    if drift_event:
        import json
        drift_details = json.dumps({
            "psi": drift_event.psi,
            "chi2": drift_event.chi2,
            "output_drift": drift_event.output_drift,
        })
    context_str = context.model_dump_json() if context else "{}"

    chat = llm.get_chat_model("action").with_structured_output(ActionPlan)
    prompt = ACTION_PROMPT.format(
        model_id=model_id,
        severity=severity,
        drift_details=drift_details,
        context=context_str,
    )

    try:
        output = await chat.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:
        raise AgentError(f"action failed: {exc}") from exc

    if not isinstance(output, ActionPlan):
        raise AgentError(f"action returned non-structured output: {type(output).__name__}")

    requires_human_review = output.action_type in PRODUCTION_TOUCHING_ACTIONS

    log.info(
        "action_done",
        model_id=model_id,
        action_type=output.action_type,
        requires_human_review=requires_human_review,
    )
    return Command(
        goto="supervisor",
        update={
            "recommended_action": output,
            "requires_human_review": requires_human_review,
        },
    )