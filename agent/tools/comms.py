"""Comms node — notifies stakeholders of the triage outcome."""
from agent.graph.state import TriageState
from agent.tasks.platform_client import notify


def run(state: TriageState) -> TriageState:
    notify(
        model_id=state["model_id"],
        decision=state["triage_decision"],
        action=state["action_taken"],
    )
    return {**state, "comms_sent": True}
