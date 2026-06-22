"""Unit tests for ``web_search``.

Pure-Python stub (no network). We drill the deterministic-result shape
plus the simulate_rate_limit hook that backs ``test_resilience``.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import ToolError
from app.workers.tasks.web_search import _RATE_LIMIT_HITS, web_search


async def test_returns_canned_results(
    settings_override: None,
    fake_redis: Any,
) -> None:
    result = await web_search(
        {"job_id": "j", "redis": fake_redis},
        {"query": "lithium recycling", "limit": 2},
    )
    assert result["query"] == "lithium recycling"
    assert len(result["results"]) == 2
    assert all("lithium recycling" in r["title"] for r in result["results"])


async def test_invalid_payload_raises_permanent(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """Missing query → PermanentToolError surfaces as ToolError after no retry."""
    with pytest.raises(ToolError):
        await web_search(
            {"job_id": "j", "redis": fake_redis},
            {"limit": 2},  # missing query
        )


async def test_simulate_rate_limit_recovers_on_retry(
    settings_override: None,
    fake_redis: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first call raises Transient; the @task wrapper retries → success."""
    _RATE_LIMIT_HITS.clear()
    # Make the test fast: zero backoff.
    monkeypatch.setenv("RETRY_BACKOFF_BASE_S", "0")
    monkeypatch.setenv("RETRY_BACKOFF_MAX_S", "0")

    from app.config import get_settings

    get_settings.cache_clear()
    try:
        result = await web_search(
            {"job_id": "rl-1", "redis": fake_redis},
            {"query": "x", "limit": 1, "simulate_rate_limit": True},
        )
        assert result["query"] == "x"
        # Verify two actual function entries: one transient + one success.
        assert _RATE_LIMIT_HITS["x"] == 2
    finally:
        get_settings.cache_clear()
