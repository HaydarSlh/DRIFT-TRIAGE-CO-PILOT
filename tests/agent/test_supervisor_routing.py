"""Tests for LangGraph supervisor routing logic."""
import pytest
from agent.graph.supervisor import route
from agent.graph.state import TriageState


def _state(**kwargs) -> TriageState:
    base: TriageState = {
        "model_id": "test-model",
        "drift_score": 0.15,
        "triage_decision": None,
        "action_taken": None,
        "comms_sent": False,
    }
    return {**base, **kwargs}


def test_routes_to_triage_when_no_decision():
    assert route(_state()) == "triage"


def test_routes_to_action_after_triage():
    assert route(_state(triage_decision="warning")) == "action"


def test_routes_to_comms_after_action():
    assert route(_state(triage_decision="warning", action_taken="task-123")) == "comms"


def test_routes_to_end_when_complete():
    from langgraph.graph import END
    assert route(_state(triage_decision="stable", action_taken="task-123", comms_sent=True)) == END
