"""Unit tests for individual agent nodes.

The pattern: instantiate a tightly-scoped ``FakeChatModel`` per test, drive
one node, assert on the returned ``Command`` AND on what was actually sent
to the LLM via ``fake.call_history``.

Each test is named for what it asserts, not the function it calls.
"""

from typing import Any

from langgraph.graph import END

from app.agents.nodes.critic import critic_node
from app.agents.nodes.researcher import ResearchOutput, researcher_node
from app.agents.nodes.supervisor import MAX_STEPS, SupervisorDecision, supervisor_node
from app.agents.nodes.writer import writer_node
from app.agents.state import Finding, ResearchState
from app.testing.fakes import FakeChatModel

# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


async def test_supervisor_routes_to_researcher_when_no_findings(
    settings_override: None, patch_chat_model: Any
) -> None:
    """First supervisor call on empty state routes to the researcher."""
    fake = patch_chat_model(
        FakeChatModel(
            by_schema={
                SupervisorDecision: SupervisorDecision(
                    next_agent="researcher", reasoning="empty findings"
                )
            }
        )
    )
    cmd = await supervisor_node({"topic": "x", "step_count": 0})

    assert cmd.goto == "researcher"
    assert cmd.update["step_count"] == 1
    # The LLM was called exactly once, with the schema we expect.
    assert len(fake.call_history) == 1
    assert fake.call_history[0].schema is SupervisorDecision


async def test_supervisor_routes_to_end_when_decision_is_done(
    settings_override: None, patch_chat_model: Any
) -> None:
    """When the LLM says 'done', the supervisor goes to END."""
    patch_chat_model(
        FakeChatModel(
            by_schema={
                SupervisorDecision: SupervisorDecision(
                    next_agent="done", reasoning="report present"
                )
            }
        )
    )
    cmd = await supervisor_node({"topic": "x", "step_count": 3, "report": "abc"})
    assert cmd.goto == END


async def test_supervisor_includes_findings_count_in_prompt(
    settings_override: None, patch_chat_model: Any
) -> None:
    """The supervisor prompt is interpolated with the current findings count."""
    fake = patch_chat_model(
        FakeChatModel(
            by_schema={
                SupervisorDecision: SupervisorDecision(
                    next_agent="critic", reasoning="findings present"
                )
            }
        )
    )
    findings = [
        Finding(title=f"F{i}", summary="s", confidence=0.5, source_hint="h") for i in range(3)
    ]
    await supervisor_node({"topic": "x", "step_count": 1, "findings": findings})
    sent = fake.call_history[0].messages_repr
    assert "findings_count: 3" in sent


async def test_supervisor_max_steps_forces_end_without_llm_call(
    settings_override: None, patch_chat_model: Any
) -> None:
    """Exceeding MAX_STEPS forces END and skips the LLM entirely."""
    fake = patch_chat_model(FakeChatModel(responses=[]))
    state: ResearchState = {"topic": "x", "step_count": MAX_STEPS}
    cmd = await supervisor_node(state)
    assert cmd.goto == END
    assert cmd.update["step_count"] == MAX_STEPS + 1
    # LLM NOT called — fail-fast guardrail.
    assert fake.call_history == []


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------


async def test_researcher_populates_findings(
    settings_override: None, patch_chat_model: Any
) -> None:
    canned = ResearchOutput(
        findings=[
            Finding(title="A", summary="alpha", confidence=0.9, source_hint="src1"),
            Finding(title="B", summary="beta", confidence=0.7, source_hint="src2"),
        ]
    )
    patch_chat_model(FakeChatModel(by_schema={ResearchOutput: canned}))

    cmd = await researcher_node({"topic": "the topic"})

    assert cmd.goto == "supervisor"
    findings = cmd.update["findings"]
    assert [f.title for f in findings] == ["A", "B"]


async def test_researcher_prompt_carries_topic(
    settings_override: None, patch_chat_model: Any
) -> None:
    canned = ResearchOutput(
        findings=[Finding(title="x", summary="y", confidence=0.5, source_hint="z")]
    )
    fake = patch_chat_model(FakeChatModel(by_schema={ResearchOutput: canned}))
    await researcher_node({"topic": "lithium battery recycling"})
    assert "lithium battery recycling" in fake.call_history[0].messages_repr


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


async def test_critic_populates_critique(
    settings_override: None, patch_chat_model: Any
) -> None:
    patch_chat_model(FakeChatModel(responses=["The findings under-cite peer-reviewed work."]))
    state: ResearchState = {
        "topic": "x",
        "findings": [Finding(title="t", summary="s", confidence=0.5, source_hint="h")],
    }
    cmd = await critic_node(state)
    assert cmd.goto == "supervisor"
    assert "peer-reviewed" in cmd.update["critique"]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


async def test_writer_populates_report(
    settings_override: None, patch_chat_model: Any
) -> None:
    patch_chat_model(FakeChatModel(responses=["# topic\n\n## Findings\n\n## Limitations\n\n..."]))
    state: ResearchState = {
        "topic": "x",
        "findings": [Finding(title="t", summary="s", confidence=0.5, source_hint="h")],
        "critique": "weak",
    }
    cmd = await writer_node(state)
    assert cmd.goto == "supervisor"
    assert "## Findings" in cmd.update["report"]
