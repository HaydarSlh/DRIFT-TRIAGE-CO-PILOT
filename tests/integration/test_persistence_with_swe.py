"""CP3's "agents as software" thesis test.

Proves that the CP2 persistence layer + CP3's testing infrastructure compose:
a graph that crashes mid-run is recoverable from its checkpoint, and a test
can drive that recovery without ever calling a real LLM.

Skips when Postgres isn't reachable (the resume path persists state).
"""

import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_graph
from app.agents.nodes.researcher import ResearchOutput
from app.agents.nodes.supervisor import SupervisorDecision
from app.agents.state import Finding
from app.testing.fakes import FakeChatModel


@pytest.mark.slow
@pytest.mark.usefixtures("settings_override")
async def test_crash_then_resume_restores_state_from_checkpoint(
    patch_chat_model, test_sessionmaker
) -> None:
    """A run that crashes after the researcher resumes from the checkpoint.

    Mechanic: the FIRST graph run uses a fake whose critic raises. We catch
    the exception, then build a SECOND graph instance that shares the same
    MemorySaver — i.e. the checkpoint is intact. We invoke with a fresh fake
    that completes successfully and assert the run finishes with the
    findings the first run already produced.
    """
    saver = MemorySaver()
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    # First run: researcher succeeds, critic raises.
    crash_decisions = iter(
        [
            SupervisorDecision(next_agent="researcher", reasoning="empty"),
            SupervisorDecision(next_agent="critic", reasoning="findings"),
        ]
    )

    class _CrashOnSecondText:
        """Plain-call fake that raises on the second `.ainvoke` (the critic)."""

        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise RuntimeError("simulated critic crash")

    crash_fake = FakeChatModel(
        by_schema={
            ResearchOutput: ResearchOutput(
                findings=[
                    Finding(title="alpha", summary="A", confidence=0.9, source_hint="h"),
                    Finding(title="beta", summary="B", confidence=0.7, source_hint="h"),
                ]
            ),
        },
    )
    crash_fake._by_schema[SupervisorDecision] = _Cycler(crash_decisions)
    # Plain-text path (the critic) raises on first call.
    original_dispatch = crash_fake._dispatch

    async def _crashing_dispatch(self_unused, *, messages, schema, kwargs):  # type: ignore[no-untyped-def]
        if schema is None:
            raise RuntimeError("simulated critic crash")
        return await original_dispatch(messages=messages, schema=schema, kwargs=kwargs)

    crash_fake._dispatch = _crashing_dispatch.__get__(crash_fake)  # type: ignore[assignment]
    patch_chat_model(crash_fake)

    graph_1 = build_graph(checkpointer=saver)
    with pytest.raises(RuntimeError, match="simulated critic crash"):
        await graph_1.ainvoke(
            {"topic": "crash-resume drill", "step_count": 0, "thread_id": thread_id},
            config=config,
        )

    # The checkpoint should now contain the findings from the researcher's
    # successful step.
    snap = await graph_1.aget_state(config)
    assert snap.values.get("findings"), "researcher's output should have been checkpointed"

    # Second run: a healthy fake. The graph picks up where it left off — we
    # don't replay the researcher because its findings are already in state.
    resume_decisions = iter(
        [
            SupervisorDecision(next_agent="critic", reasoning="findings present"),
            SupervisorDecision(next_agent="writer", reasoning="critique present"),
            SupervisorDecision(next_agent="done", reasoning="report present"),
        ]
    )
    resume_fake = FakeChatModel(
        by_schema={},
        responses=[
            "Critique: weak citations.",
            "# crash-resume drill\n\n## Findings\n\n## Limitations\n\nResumed run.",
        ],
    )
    resume_fake._by_schema[SupervisorDecision] = _Cycler(resume_decisions)
    patch_chat_model(resume_fake)

    graph_2 = build_graph(checkpointer=saver)
    final = await graph_2.ainvoke(None, config=config)

    assert final["report"], "resumed run should produce a final report"
    assert "Resumed run" in final["report"]
    # The researcher's findings persisted across the crash.
    assert [f.title for f in final["findings"]] == ["alpha", "beta"]


class _Cycler:
    def __init__(self, source) -> None:
        self._source = source

    def next(self):
        try:
            return next(self._source)
        except StopIteration as exc:
            raise AssertionError("decision sequence exhausted") from exc
