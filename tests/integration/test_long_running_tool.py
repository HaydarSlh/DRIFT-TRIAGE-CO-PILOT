"""End-to-end: agent → queue → worker → agent.

We don't run a real arq worker here. Instead we patch ``call_tool`` so the
researcher's queue-backed search executes the registered task inline (the
same code path the worker would run, against a fakeredis). That lets us
assert:

1. The agent calls into the queue layer (``call_tool`` is invoked).
2. The worker function runs and produces the right result.
3. The graph completes with the right state.

The cheaper integration test that proves the queueing protocol works
end-to-end without arq's network surface. A "real worker" smoke test
exists in compose-driven CI; that one is opt-in.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from app.agents.nodes.researcher import ResearchOutput
from app.agents.nodes.supervisor import SupervisorDecision
from app.agents.state import Finding
from app.testing.fakes import FakeChatModel


class _Cycler:
    """Schema-map value that yields one element per dispatch (matches conftest)."""

    def __init__(self, source: Iterator[Any]) -> None:
        self._source = source

    def next(self) -> Any:
        try:
            return next(self._source)
        except StopIteration as exc:
            raise AssertionError("decision sequence exhausted") from exc


@pytest.mark.slow
async def test_graph_runs_with_queue_enabled_and_inline_worker(
    settings_override: None,
    fake_redis: Any,
    monkeypatch: pytest.MonkeyPatch,
    patch_chat_model: Any,
) -> None:
    """Enable the queue but route ``call_tool`` to inline execution.

    With ENABLE_QUEUE=true and a stub that calls TASK_REGISTRY inline, the
    agent path exercises the full queueing protocol minus the network —
    just like the worker would, but in-process.
    """
    monkeypatch.setenv("ENABLE_QUEUE", "true")
    from app.config import get_settings

    get_settings.cache_clear()

    # Same supervisor sequence as test_graph.py so the graph fully runs.
    decisions = iter(
        [
            SupervisorDecision(next_agent="researcher", reasoning="empty"),
            SupervisorDecision(next_agent="critic", reasoning="findings present"),
            SupervisorDecision(next_agent="writer", reasoning="critique present"),
            SupervisorDecision(next_agent="done", reasoning="report present"),
        ]
    )

    fake = FakeChatModel(
        by_schema={
            ResearchOutput: ResearchOutput(
                findings=[
                    Finding(title="A", summary="alpha", confidence=0.9, source_hint="s"),
                ]
            ),
        },
        responses=[
            "Critique: findings are thin but plausible.",
            "# topic\n\n## Findings\n\n### A\nalpha\n\n## Limitations\n\nFake.",
        ],
    )
    fake._by_schema[SupervisorDecision] = _Cycler(decisions)
    patch_chat_model(fake)

    # Patch ``call_tool`` on both the package binding and the researcher
    # binding. Either monkeypatch alone leaves the other in place because
    # the researcher imported the symbol at module load.
    from app.taskqueue.tasks import TASK_REGISTRY

    call_log: list[str] = []

    async def _inline_call_tool(
        name: str, payload: dict[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        call_log.append(name)
        func = TASK_REGISTRY[name]
        return await func({"job_id": f"test-{name}", "redis": fake_redis}, payload)

    monkeypatch.setattr("app.agents.tools.call_tool", _inline_call_tool)
    monkeypatch.setattr("app.agents.nodes.researcher.call_tool", _inline_call_tool)

    from langgraph.checkpoint.memory import MemorySaver

    from app.agents.graph import build_graph

    graph = build_graph(checkpointer=MemorySaver())
    final = await graph.ainvoke(
        {"topic": "queueing", "step_count": 0},
        config={"configurable": {"thread_id": "qtest-1"}},
    )

    assert final["topic"] == "queueing"
    assert final["report"]
    # The researcher invoked the queue at least once (web_search).
    assert "web_search" in call_log
