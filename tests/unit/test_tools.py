"""Unit tests for pure utility functions ("tool-shaped" code).

CP3's agents do not yet call external tools — but the code already has
several small pure functions that benefit from the same isolation: prompt
formatters, summarizers, deterministic helpers. These are the easiest place
to demonstrate the test pyramid's bottom layer.

When tools land in a future checkpoint (web search, file readers, etc.),
their tests live here too.
"""

import pytest

from app.agents.nodes.critic import _format_findings as format_findings_critic
from app.agents.nodes.writer import _format_findings as format_findings_writer
from app.agents.state import Finding

# ---------------------------------------------------------------------------
# _format_findings
# ---------------------------------------------------------------------------
#
# Identical implementations live in critic.py and writer.py for now (so each
# node is self-contained). We assert both behave the same — that gives us
# permission to deduplicate later without breaking either node.


@pytest.mark.parametrize(
    "format_fn",
    [format_findings_critic, format_findings_writer],
    ids=["critic", "writer"],
)
def test_format_findings_returns_no_findings_marker_when_empty(format_fn) -> None:
    assert format_fn([]) == "(no findings)"


@pytest.mark.parametrize(
    "format_fn",
    [format_findings_critic, format_findings_writer],
    ids=["critic", "writer"],
)
def test_format_findings_renders_each_on_its_own_line(format_fn) -> None:
    findings = [
        Finding(title="A", summary="alpha", confidence=0.91, source_hint="src1"),
        Finding(title="B", summary="beta", confidence=0.45, source_hint="src2"),
    ]
    rendered = format_fn(findings)
    lines = rendered.splitlines()
    assert len(lines) == 2
    assert "A" in lines[0]
    assert "alpha" in lines[0]
    assert "0.91" in lines[0]
    assert "B" in lines[1]
    assert "0.45" in lines[1]


@pytest.mark.parametrize(
    "format_fn",
    [format_findings_critic, format_findings_writer],
    ids=["critic", "writer"],
)
def test_format_findings_includes_confidence_with_two_decimals(format_fn) -> None:
    """Confidence is formatted to 2dp — guard against drift to 0.5 or 0.500."""
    rendered = format_fn(
        [Finding(title="t", summary="s", confidence=0.5, source_hint="h")]
    )
    assert "confidence=0.50" in rendered


# ---------------------------------------------------------------------------
# State / config sanity
# ---------------------------------------------------------------------------


def test_finding_rejects_confidence_outside_unit_interval() -> None:
    """Pydantic validation must reject impossible confidences."""
    with pytest.raises(ValueError):
        Finding(title="t", summary="s", confidence=1.5, source_hint="h")
    with pytest.raises(ValueError):
        Finding(title="t", summary="s", confidence=-0.1, source_hint="h")


def test_settings_is_deterministic_property(settings_override: None) -> None:
    """The is_deterministic helper combines both env-var aliases."""
    from app.config import get_settings

    s = get_settings()
    # The autouse fixture set TEST_DETERMINISTIC=true.
    assert s.is_deterministic is True
