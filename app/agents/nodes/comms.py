"""Comms node — drafts a notification and dispatches Production actions.

On entry, if ``requires_human_review`` is True (i.e., the action is
Production-touching: retrain_shadow or rollback), the comms node calls
``interrupt(...)`` BEFORE calling the action tool. LangGraph saves a
checkpoint and the graph pauses; the reviewer's response is fed back via
``Command(resume=...)``.

On resume, if approved, the comms node dispatches the action through the
queue layer. If rejected, it ends without dispatching.

Actions that don't touch Production (``none`` and ``replay_test``) complete
without pausing for human review.
"""

from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt

from app.agents import llm
from app.agents.prompts.comms import COMMS_PROMPT
from app.agents.state import ActionPlan, TriageState
from app.agents.tools import call_tool
from app.config import get_settings
from app.core.errors import AgentError
from app.core.logging import get_logger

log = get_logger(__name__)


def _format_action(action: ActionPlan) -> str:
    """Render an ActionPlan as a short human-readable description."""
    lines = [f"Action: {action.action_type}"]
    if action.target_version:
        lines.append(f"Target version: {action.target_version}")
    lines.append(f"Justification: {action.justification}")
    return "\n".join(lines)


async def comms_node(state: TriageState) -> Command[Literal["supervisor"]]:
    """Draft a notification and, if needed, dispatch the remediation action.

    Args:
        state: Current graph state. Reads ``drift_event``, ``severity``,
            ``recommended_action``, ``requires_human_review``. May read
            ``human_feedback`` after a review interrupt resumes.

    Returns:
        ``Command(goto="supervisor", update={"comms_draft": ...})``.

    Raises:
        AgentError: If the LLM call fails.
    """
    drift_event = state.get("drift_event")
    severity = state.get("severity", "UNKNOWN")
    action = state.get("recommended_action")
    model_id = drift_event.model_id if drift_event else "unknown"

    feedback_context = state.get("human_feedback")

    # HIL interrupt: pause before any Production-touching action.
    if state.get("requires_human_review") and not feedback_context:
        review_payload: dict[str, Any] = {
            "prompt": (
                f"Approve the proposed remediation for model {model_id}? "
                f"Severity: {severity}. Action: {action.action_type if action else 'N/A'}."
            ),
            "model_id": model_id,
            "severity": severity,
            "action_plan": action.model_dump() if action else {},
        }
        log.info("comms_interrupt_for_review", model_id=model_id)
        decision = interrupt(review_payload)
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else False
        feedback_context = (
            decision.get("feedback", "") if isinstance(decision, dict) else str(decision)
        )
        if not approved:
            log.info("comms_review_rejected", model_id=model_id, feedback=feedback_context)
            draft = (
                f"Drift alert for {model_id} (severity: {severity}). "
                f"Proposed action was rejected by reviewer. "
                f"Reviewer note: {feedback_context or '(none)'}"
            )
            return Command(
                goto="supervisor",
                update={"comms_draft": draft, "human_feedback": feedback_context},
            )

    # If approved (or never required review), dispatch Production-touching actions.
    if action and action.action_type in ("retrain_shadow", "rollback"):
        action_type = action.action_type
        payload = {
            "model_id": model_id,
            "action": action_type,
        }
        if action.target_version:
            payload["target_version"] = action.target_version

        settings = get_settings()
        if settings.enable_queue or settings.enable_queue_fallback:
            try:
                result = await call_tool(
                    action_type,
                    payload,
                    idempotency_key=f"{state.get('thread_id', 'unknown')}:{action_type}",
                )
                log.info("comms_action_dispatched", action_type=action_type, result=result)
            except Exception as exc:
                log.warning("comms_action_dispatch_failed", action_type=action_type, error=str(exc))

    # Draft notification.
    chat = llm.get_chat_model("comms")
    action_str = _format_action(action) if action else "none"
    prompt = COMMS_PROMPT.format(
        model_id=model_id,
        severity=severity,
        action_description=action_str,
    )
    if feedback_context:
        prompt += f"\n\nReviewer feedback to incorporate: {feedback_context}\n"

    try:
        response = await chat.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:
        raise AgentError(f"comms failed: {exc}") from exc

    draft = response.content if hasattr(response, "content") else str(response)
    if not isinstance(draft, str):
        draft = str(draft)

    log.info("comms_done", model_id=model_id, draft_chars=len(draft))
    return Command(
        goto="supervisor",
        update={"comms_draft": draft, "human_feedback": feedback_context},
    )