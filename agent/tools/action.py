"""Action node — dispatches remediation task to the worker queue."""
from agent.graph.state import TriageState
from agent.tasks.queue_dispatch import dispatch


def run(state: TriageState) -> TriageState:
    task_id = dispatch(
        model_id=state["model_id"],
        decision=state["triage_decision"],
    )
    return {**state, "action_taken": task_id}
