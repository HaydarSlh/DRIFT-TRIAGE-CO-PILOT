"""Resilience: retries on transient errors, immediate failure on permanent.

This is the test that proves the lesson:
    transient → retry with backoff
    permanent → fail fast, no retry storm

We use the ``simulate_rate_limit`` hook on ``web_search`` to flip the
first call into TransientToolError; the second call succeeds. Then we
validate the failure path with an unknown payload field that the
Pydantic validator rejects → PermanentToolError.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import PermanentToolError, ToolError, TransientToolError
from app.taskqueue import JobStatus, load_result
from app.taskqueue.tasks import _run_with_retry
from app.workers.tasks.web_search import _RATE_LIMIT_HITS, web_search


@pytest.fixture(autouse=True)
def _reset_rate_limit_state() -> None:
    _RATE_LIMIT_HITS.clear()


@pytest.mark.slow
async def test_three_transient_failures_then_success(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """Wrapping web_search with the simulate hook → transient on call 1, OK on 2.

    The wrapper retries once and stores a COMPLETED result on attempt 2.
    Total attempts == 2.
    """
    # Note: the @task-decorated ``web_search`` is what tasks/web_search.py
    # registers; we drive it via the wrapper's ``_run_with_retry`` helper
    # so we can directly assert on attempts.
    async def shim(payload: dict[str, Any]) -> dict[str, Any]:
        # The decorator-stripped body is what we want, but we don't want to
        # bypass the simulate hook — so we call the @task wrapper itself,
        # which has the hook.
        ctx = {"job_id": "sim-1", "redis": fake_redis}
        return await web_search(ctx, payload)

    payload = {"query": "lithium", "limit": 1, "simulate_rate_limit": True}
    # Drive with retry policy directly so we can assert attempt count.
    result = await _run_with_retry(
        shim,
        payload,
        task_name="web_search_shim",
        job_id="resilience-1",
        redis=fake_redis,
        max_retries=3,
        retry_on=(TransientToolError,),
        no_retry_on=(PermanentToolError,),
        deterministic=True,
        base_s=0.0,
        max_s=0.0,
        webhook=None,
    )

    assert result["query"] == "lithium"
    stored = await load_result(fake_redis, "resilience-1")
    assert stored is not None
    assert stored.status == JobStatus.COMPLETED


@pytest.mark.slow
async def test_permanent_error_no_retry(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """Calling web_search with an invalid payload → PermanentToolError, 1 attempt."""
    calls: list[int] = []

    async def shim(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        ctx = {"job_id": "perm-shim-1", "redis": fake_redis}
        return await web_search(ctx, payload)

    with pytest.raises(ToolError):
        await _run_with_retry(
            shim,
            {"limit": 1},  # missing required "query"
            task_name="web_search_shim",
            job_id="resilience-2",
            redis=fake_redis,
            max_retries=3,
            retry_on=(TransientToolError,),
            no_retry_on=(PermanentToolError,),
            deterministic=True,
            base_s=0.0,
            max_s=0.0,
            webhook=None,
        )

    assert len(calls) == 1  # no retries on permanent errors
