"""Triage node — classifies drift severity and sets triage_decision."""
from agent.graph.state import TriageState


def run(state: TriageState) -> TriageState:
    score = state["drift_score"]
    if score >= 0.25:
        decision = "critical"
    elif score >= 0.10:
        decision = "warning"
    else:
        decision = "stable"
    return {**state, "triage_decision": decision}
