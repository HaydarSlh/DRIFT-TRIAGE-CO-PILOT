"""Integration test: drive the compiled graph end-to-end with a fake LLM.

Asserts that:
- the graph reaches a terminal state populated with findings + critique + report
- NO real LLM calls were made (we count fake invocations)
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_graph
from app.agents.nodes.researcher import ResearchOutput
from app.agents.nodes.supervisor import SupervisorDecision
from app.agents.state import Finding
from app.testing.fakes import FakeChatModel


@pytest.mark.usefixtures("settings_override")
async def test_graph_runs_end_to_end_with_fake(patch_chat_model) -> None:
    """One supervisor → researcher → supervisor → critic → supervisor → writer → END."""
    # Build a fake whose ordering exactly matches the expected supervisor cycle.
    decisions = iter(
        [
            SupervisorDecision(next_agent="researcher", reasoning="empty"),
            SupervisorDecision(next_agent="critic", reasoning="findings present"),
            SupervisorDecision(next_agent="writer", reasoning="critique present"),
            SupervisorDecision(next_agent="done", reasoning="report present"),
        ]
    )

    # Stateful schema-map: each call to ``with_structured_output(SupervisorDecision)``
    # picks the next decision. We override the dispatcher locally for clarity
    # rather than reusing the conftest cycler.
    fake = FakeChatModel(
        by_schema={
            ResearchOutput: ResearchOutput(
                findings=[
                    Finding(title="alpha", summary="A", confidence=0.9, source_hint="h"),
                    Finding(title="beta", summary="B", confidence=0.7, source_hint="h"),
                ]
            ),
        },
        responses=[
            "The findings are reasonable but cite no sources.",
            "# topic\n\n## Findings\n\n### alpha\nA\n\n## Limitations\n\nLow confidence.",
        ],
    )
    # Inject the iterator into the by_schema map under SupervisorDecision.
    fake._by_schema[SupervisorDecision] = _Cycler(decisions)

    patch_chat_model(fake)

    graph = build_graph(checkpointer=MemorySaver())
    final = await graph.ainvoke(
        {"topic": "test topic for graph integration", "step_count": 0},
        config={"configurable": {"thread_id": "graph-int-1"}},
    )

    assert final["topic"] == "test topic for graph integration"
    assert len(final["findings"]) == 2
    assert "no sources" in final["critique"]
    assert "## Findings" in final["report"]
    assert final["step_count"] == 4

    # Every supervisor decision (4) plus researcher (1) plus critic (1) plus writer (1) = 7
    assert len(fake.call_history) == 7


class _Cycler:
    """Schema-map value that yields one element per dispatch."""

    def __init__(self, source) -> None:
        self._source = source

    def next(self):  # the cycler-aware dispatch in conftest looks for .next()
        try:
            return next(self._source)
        except StopIteration as exc:
            raise AssertionError("supervisor decision sequence exhausted") from exc
