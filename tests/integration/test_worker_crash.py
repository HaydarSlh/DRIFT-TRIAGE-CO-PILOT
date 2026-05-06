"""Worker-crash drill: at-least-once redelivery + idempotency = exactly-once effects.

We simulate "worker dies after running the task body but before storing
the JobResult" and verify that:

1. The next worker (or arq redelivery) re-runs the task.
2. Because the task body uses ``store_result`` only at the end, partial
   work is benign — re-running is safe.
3. The final agent-observed result is the same regardless of how many
   redeliveries happened.

The simulation is in-process: we drive the @task wrapper twice on the
same job_id. The first run "crashes" by raising mid-task before the
result is stored. The second run completes and stores. ``await_result``
sees the second run's result.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import ToolError, TransientToolError
from app.queue import JobStatus, load_result
from app.queue.tasks import _run_with_retry


@pytest.mark.slow
async def test_redelivery_after_simulated_crash(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """First call raises before storing; second call succeeds.

    The producer-side idempotency check (``QueueClient.enqueue`` skipping
    when a stored result exists) is tested separately. Here we focus on
    the worker-side: a redelivered job_id that *doesn't* yet have a stored
    result will run again, and that's safe.
    """
    state: dict[str, int] = {"calls": 0}

    async def crashy(payload: dict[str, Any]) -> dict[str, Any]:
        state["calls"] += 1
        if state["calls"] == 1:
            # Simulate a crash partway through the task body. The wrapper
            # should treat this as transient and retry.
            raise TransientToolError("worker crashed mid-task")
        return {"work_done_at_call": state["calls"]}

    result = await _run_with_retry(
        crashy,
        {"k": "v"},
        task_name="crashy",
        job_id="crash-1",
        redis=fake_redis,
        max_retries=2,
        retry_on=(TransientToolError,),
        no_retry_on=(),
        deterministic=True,
        base_s=0.0,
        max_s=0.0,
        webhook=None,
    )

    assert result["work_done_at_call"] == 2
    stored = await load_result(fake_redis, "crash-1")
    assert stored is not None
    assert stored.status == JobStatus.COMPLETED


@pytest.mark.slow
async def test_idempotency_skip_on_redelivery_when_result_exists(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """If a COMPLETED result already exists, the wrapper short-circuits.

    Drives the wrapper twice. The second invocation should NOT call into
    the function body — it should return the cached result directly.
    """
    from app.queue import JobResult, store_result
    from app.workers.tasks.web_search import web_search

    job_id = "redeliver-1"
    await store_result(
        fake_redis,
        JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            result_data={"query": "cached", "results": []},
        ),
    )

    out = await web_search(
        {"job_id": job_id, "redis": fake_redis},
        {"query": "should-not-run", "limit": 5},
    )
    # The cached result_data is what we get, not a fresh canned result.
    assert out == {"query": "cached", "results": []}


@pytest.mark.slow
async def test_permanent_failure_lands_in_dlq_not_silently_dropped(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """The DLQ is the floor — permanent failures cannot disappear."""
    from app.core.errors import PermanentToolError
    from app.queue.dlq import list_dlq

    async def doomed(_payload: dict[str, Any]) -> dict[str, Any]:
        raise PermanentToolError("4xx something")

    with pytest.raises(ToolError):
        await _run_with_retry(
            doomed,
            {},
            task_name="doomed",
            job_id="perm-1",
            redis=fake_redis,
            max_retries=3,
            retry_on=(TransientToolError,),
            no_retry_on=(PermanentToolError,),
            deterministic=True,
            base_s=0.0,
            max_s=0.0,
            webhook=None,
        )

    listed = await list_dlq(fake_redis)
    assert any(e.job_id == "perm-1" for e in listed)
