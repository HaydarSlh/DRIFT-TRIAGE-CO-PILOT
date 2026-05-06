"""Idempotency tests.

Two scenarios:

1. **Same idempotency_key → same job_id, only one execution.**
   Producer calls ``enqueue`` twice with the same key. The second call
   returns the same job_id (deterministic hash) AND does not actually
   submit a new job to arq.

2. **No idempotency_key → two job_ids, two executions.**
   Producer calls ``enqueue`` twice with the same payload. arq generates
   distinct ids; both make it onto the wire.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.queue import JobResult, JobStatus, QueueClient, store_result
from tests.queue.conftest import StubArqPool


async def test_same_idempotency_key_returns_same_job_id(
    settings_override: None,
    fake_redis: Any,
) -> None:
    arq = StubArqPool()
    qc = QueueClient(arq_pool=arq, redis=fake_redis, settings=get_settings())

    job_id_1 = await qc.enqueue("scrape_url", {"url": "https://x"}, idempotency_key="x")
    job_id_2 = await qc.enqueue("scrape_url", {"url": "https://x"}, idempotency_key="x")

    assert job_id_1 == job_id_2
    # arq saw only one new submission; the second was deduped.
    assert len(arq.enqueued) == 1


async def test_idempotency_skip_when_result_already_stored(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """If a JobResult exists for the deterministic id, don't even call arq."""
    arq = StubArqPool()
    qc = QueueClient(arq_pool=arq, redis=fake_redis, settings=get_settings())
    job_id_pre = QueueClient._deterministic_job_id("scrape_url", "x")

    # Pre-populate a completed result.
    await store_result(
        fake_redis,
        JobResult(
            job_id=job_id_pre,
            status=JobStatus.COMPLETED,
            result_data={"content": "cached"},
        ),
    )

    job_id = await qc.enqueue("scrape_url", {"url": "https://x"}, idempotency_key="x")

    assert job_id == job_id_pre
    assert arq.enqueued == []  # the producer never even submitted


async def test_no_idempotency_key_means_distinct_jobs(
    settings_override: None,
    fake_redis: Any,
) -> None:
    """Without an idempotency_key, two calls = two submissions."""
    arq = StubArqPool()
    qc = QueueClient(arq_pool=arq, redis=fake_redis, settings=get_settings())

    await qc.enqueue("web_search", {"query": "a"})
    await qc.enqueue("web_search", {"query": "a"})

    assert len(arq.enqueued) == 2
