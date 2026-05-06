"""Trajectory snapshot tests.

These are the ones a one-line prompt change is most likely to break — and
that's the point. A change to ``app/agents/prompts/supervisor.py`` that
shifts which worker the supervisor calls in what order will fail this test
*before* it ever ships.

Updating the snapshots is a deliberate act:

    uv run pytest tests/snapshot/ --snapshot-update

Do NOT blind-update. Read the diff. If the new trajectory looks worse, the
prompt change is wrong. If it looks better, ship the update with a one-line
note in ``app/agents/prompts/CHANGELOG.md``.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_graph
from app.agents.nodes.researcher import ResearchOutput
from app.agents.nodes.supervisor import SupervisorDecision
from app.agents.state import Finding
from app.testing.fakes import FakeChatModel

pytestmark = pytest.mark.snapshot


def _build_happy_path_fake() -> FakeChatModel:
    """A fake that lets the supervisor → researcher → critic → writer → done flow run."""
    decisions = iter(
        [
            SupervisorDecision(next_agent="researcher", reasoning="empty"),
            SupervisorDecision(next_agent="critic", reasoning="findings"),
            SupervisorDecision(next_agent="writer", reasoning="critique"),
            SupervisorDecision(next_agent="done", reasoning="report"),
        ]
    )
    fake = FakeChatModel(
        by_schema={
            ResearchOutput: ResearchOutput(
                findings=[
                    Finding(title="alpha", summary="A", confidence=0.9, source_hint="h"),
                    Finding(title="beta", summary="B", confidence=0.7, source_hint="h"),
                    Finding(title="gamma", summary="C", confidence=0.5, source_hint="h"),
                ]
            ),
        },
        responses=[
            "Critique: well-sourced but conservative.",
            "# topic\n\n## Findings\n\n### alpha\nA\n\n## Limitations\n\nLow confidence on gamma.",
        ],
    )
    fake._by_schema[SupervisorDecision] = _Cycler(decisions)
    return fake


@pytest.mark.usefixtures("settings_override")
async def test_simple_topic_trajectory(patch_chat_model, snapshot, trajectory_recorder) -> None:
    """The trajectory of a basic topic is byte-for-byte stable run-to-run."""
    patch_chat_model(_build_happy_path_fake())

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "snap-simple"}}
    async for _ in trajectory_recorder.run(
        graph, {"topic": "the economic impact of remote work", "step_count": 0}, config=config
    ):
        pass

    assert trajectory_recorder.trajectory == snapshot


@pytest.mark.usefixtures("settings_override")
async def test_complex_topic_trajectory(
    patch_chat_model, snapshot, trajectory_recorder
) -> None:
    """A multi-clause topic produces the same trajectory shape."""
    patch_chat_model(_build_happy_path_fake())

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "snap-complex"}}
    async for _ in trajectory_recorder.run(
        graph,
        {
            "topic": (
                "the impact of widespread agent-based research tooling on "
                "the cost structure of professional analyst firms"
            ),
            "step_count": 0,
        },
        config=config,
    ):
        pass

    assert trajectory_recorder.trajectory == snapshot


class _Cycler:
    def __init__(self, source) -> None:
        self._source = source

    def next(self):
        try:
            return next(self._source)
        except StopIteration as exc:
            raise AssertionError("decision sequence exhausted") from exc
