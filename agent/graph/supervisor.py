"""LangGraph supervisor — state definition, node routing, graph construction."""
from __future__ import annotations

from langgraph.graph import StateGraph, END
from agent.graph.state import TriageState
from agent.tools import triage, action, comms


def route(state: TriageState) -> str:
    if state["triage_decision"] is None:
        return "triage"
    if state["action_taken"] is None:
        return "action"
    if not state["comms_sent"]:
        return "comms"
    return END


def run_graph(event: dict) -> TriageState:
    initial: TriageState = {**event, "triage_decision": None, "action_taken": None, "comms_sent": False}
    return graph.invoke(initial)


builder = StateGraph(TriageState)
builder.add_node("triage", triage.run)
builder.add_node("action", action.run)
builder.add_node("comms", comms.run)
builder.set_entry_point("triage")
builder.add_conditional_edges("triage", route)
builder.add_conditional_edges("action", route)
builder.add_conditional_edges("comms", route)

graph = builder.compile()
