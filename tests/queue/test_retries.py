"""Retry policy tests.

Three behaviors to drill:

1. A function that fails N times then succeeds completes after N retries.
2. A function that always fails ends in the DLQ after MAX_RETRIES + 1 attempts.
3. ``PermanentToolError`` short-circuits the retry policy — one attempt,
   then DLQ (no exponential delay, no waiting).

We don't actually wait for backoff seconds — ``Settings.retry_backoff_base_s``
is overridden to 0 in the deterministic test mode and we monkeypatch
``asyncio.sleep`` to a no-op so tests stay fast.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import PermanentToolError, ToolError, TransientToolError
from app.taskqueue import JobStatus, load_result
from app.taskqueue.dlq import list_dlq
from app.taskqueue.tasks import _run_with_retry


@pytest.fixture(autouse=True)
def _zero_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make tenacity's exponential-jitter wait return 0 so tests don't sleep."""
    import tenacity.wait

    monkeypatch.setattr(
        tenacity.wait.wait_exponential_jitter, "__call__", lambda self, retry_state: 0.0
    )


async def test_retries_then_succeeds(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """Two TransientToolError, then success — total 3 attempts."""
    calls: list[int] = []

    async def flaky(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        if len(calls) < 3:
            raise TransientToolError("flaky")
        return {"ok": True, "calls": len(calls)}

    result = await _run_with_retry(
        flaky,
        {"k": "v"},
        task_name="flaky",
        job_id="job-1",
        redis=fake_redis,
        max_retries=3,
        retry_on=(TransientToolError,),
        no_retry_on=(PermanentToolError,),
        deterministic=True,
        base_s=0.0,
        max_s=0.0,
        webhook=None,
    )

    assert result["ok"] is True
    assert result["calls"] == 3
    stored = await load_result(fake_redis, "job-1")
    assert stored is not None
    assert stored.status == JobStatus.COMPLETED
    assert stored.attempts == 3


async def test_always_fails_ends_in_dlq(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """4 attempts (initial + 3 retries) then DLQ."""

    async def doomed(_payload: dict[str, Any]) -> dict[str, Any]:
        raise TransientToolError("never works")

    with pytest.raises(ToolError):
        await _run_with_retry(
            doomed,
            {},
            task_name="doomed",
            job_id="job-2",
            redis=fake_redis,
            max_retries=3,
            retry_on=(TransientToolError,),
            no_retry_on=(PermanentToolError,),
            deterministic=True,
            base_s=0.0,
            max_s=0.0,
            webhook=None,
        )

    stored = await load_result(fake_redis, "job-2")
    assert stored is not None
    assert stored.status == JobStatus.DEAD_LETTER

    dlq_entries = await list_dlq(fake_redis)
    assert len(dlq_entries) == 1
    assert dlq_entries[0].job_id == "job-2"
    assert dlq_entries[0].attempts == 4


async def test_permanent_error_short_circuits_retries(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """PermanentToolError → one attempt, no retries, straight to DLQ."""
    calls: list[int] = []

    async def auth_error(_payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        raise PermanentToolError("invalid credentials")

    with pytest.raises(ToolError):
        await _run_with_retry(
            auth_error,
            {},
            task_name="auth_error",
            job_id="job-3",
            redis=fake_redis,
            max_retries=3,
            retry_on=(TransientToolError,),
            no_retry_on=(PermanentToolError,),
            deterministic=True,
            base_s=0.0,
            max_s=0.0,
            webhook=None,
        )

    assert len(calls) == 1  # no retries
    dlq_entries = await list_dlq(fake_redis)
    assert len(dlq_entries) == 1
    assert "invalid credentials" in dlq_entries[0].error
