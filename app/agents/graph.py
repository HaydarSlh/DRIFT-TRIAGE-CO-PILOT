"""LangGraph supervisor wiring.

Topology: the supervisor is the entry point and the only router.
the nodes are now triage / action / comms instead of
researcher / critic / writer. The dynamic interrupt fires from inside
``comms_node``: when state's ``requires_human_review`` is True (which happens
when the action agent chooses a Production-touching action like retrain_shadow
or rollback), the comms agent calls ``interrupt(...)`` instead of dispatching
the action, and the graph pauses with a checkpoint. The reviewer's response
is fed back via ``Command(resume=...)`` on the next ``ainvoke``.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.action import action_node
from app.agents.nodes.comms import comms_node
from app.agents.nodes.supervisor import supervisor_node
from app.agents.nodes.triage import triage_node
from app.agents.state import TriageState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Build and compile the supervisor graph.

    Args:
        checkpointer: A LangGraph checkpoint saver. ``None`` is allowed for
            unit tests, but production runs MUST pass a real saver because the
            human-review interrupt cannot pause without one.

    Returns:
        A compiled ``StateGraph`` ready to be invoked.
    """
    graph: StateGraph = StateGraph(TriageState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("triage", triage_node)
    graph.add_node("action", action_node)
    graph.add_node("comms", comms_node)
    graph.add_edge(START, "supervisor")
    return graph.compile(checkpointer=checkpointer)


def build_graph_for_studio() -> CompiledStateGraph:
    """LangGraph Studio entry point.

    Studio inspects the graph topology before any infrastructure is up, so we hand
    it an in-memory checkpointer rather than dialing Redis. Running graphs from
    inside Studio uses this same in-memory saver — fine for visual exploration,
    not for durable runs.
    """
    from langgraph.checkpoint.memory import MemorySaver

    return build_graph(checkpointer=MemorySaver())