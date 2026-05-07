"""Dead-letter queue.

Jobs that exhaust all retries land here. The DLQ is a Redis list at
``dlq:agent_tools``; each entry is a JSON blob carrying enough metadata that
an operator can reproduce the failure outside the queue:

- the original task name and payload
- the final error (stringified exception)
- attempt count and timestamps
- the job_id (so a producer asking ``await_result(job_id)`` still sees a
  ``DEAD_LETTER`` JobResult)

We never silently drop a poisoned job. At minimum a ``structlog.warning``
fires; if ``DLQ_ALERT_WEBHOOK`` is configured a JSON POST goes out too. Loss
of *that* webhook would be a CP-Day-2 problem; for the bootcamp the principle
is "never silent."

Admin helpers (``list_dlq``, ``requeue_from_dlq``, ``purge_dlq``) drive the
``app.cli.queue_admin`` commands and the ``/jobs/dlq/*`` admin endpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from app.core.logging import get_logger
from app.taskqueue.results import JobResult, JobStatus, store_result

if TYPE_CHECKING:
    from app.taskqueue.client import QueueClient

# We type the ``redis`` argument as ``Any`` rather than ``redis.asyncio.Redis``
# because redis-py's stubs use a sync/async dual signature that resolves to
# ``Awaitable[X] | X``; mypy can't narrow that automatically and the noise
# would dominate this file. The behavior is checked end-to-end in
# tests/queue/test_dlq.py.

DLQ_KEY = "dlq:agent_tools"
log = get_logger(__name__)


@dataclass
class DLQEntry:
    """One entry in the DLQ Redis list."""

    job_id: str
    task_name: str
    payload: dict[str, Any]
    error: str
    attempts: int
    enqueued_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_json(self) -> str:
        return json.dumps(
            {
                "job_id": self.job_id,
                "task_name": self.task_name,
                "payload": self.payload,
                "error": self.error,
                "attempts": self.attempts,
                "enqueued_at": self.enqueued_at,
            },
            default=str,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> DLQEntry:
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls(**json.loads(raw))


async def push_to_dlq(
    redis: Any,
    *,
    job_id: str,
    task_name: str,
    payload: dict[str, Any],
    error: str,
    attempts: int,
    webhook_url: str | None = None,
    result_ttl: int = 3600,
) -> DLQEntry:
    """Move a permanently-failed job into the DLQ.

    Writes three things in order:
        1. The DLQ entry (LPUSH so newest-first reads work naturally).
        2. A ``DEAD_LETTER`` JobResult so producers awaiting the job_id
           observe the terminal state without checking the DLQ list.
        3. (optional) A webhook POST for external alerting.

    Args:
        redis: Connected redis client.
        job_id: The arq job id that failed permanently.
        task_name: Worker task name (e.g. ``scrape_url``).
        payload: The original input payload — kept verbatim so an admin can
            requeue without reconstructing it.
        error: Stringified final exception. Don't log a stack trace here;
            the worker will already have logged that in detail.
        attempts: Total attempts including the failing one.
        webhook_url: Optional URL to POST a JSON alert to.
        result_ttl: TTL on the JobResult key. Match this to ``QueueClient``
            settings so producers get a consistent view.

    Returns:
        The created DLQ entry.
    """
    entry = DLQEntry(
        job_id=job_id,
        task_name=task_name,
        payload=payload,
        error=error,
        attempts=attempts,
    )

    await redis.lpush(DLQ_KEY, entry.to_json())
    await store_result(
        redis,
        JobResult(
            job_id=job_id,
            status=JobStatus.DEAD_LETTER,
            error=error,
            attempts=attempts,
        ),
        ttl=result_ttl,
    )

    log.warning(
        "dlq_entry_added",
        job_id=job_id,
        task_name=task_name,
        attempts=attempts,
        error=error,
    )

    if webhook_url:
        await _fire_webhook(webhook_url, entry)

    return entry


async def _fire_webhook(url: str, entry: DLQEntry) -> None:
    """POST a JSON payload to ``url``. Never raises; we already have the entry."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json=json.loads(entry.to_json()))
    except Exception as exc:
        log.warning("dlq_webhook_failed", url=url, error=str(exc))


async def list_dlq(redis: Any, *, limit: int = 100) -> list[DLQEntry]:
    """Return up to ``limit`` newest DLQ entries (newest first)."""
    raw_entries = await redis.lrange(DLQ_KEY, 0, limit - 1)
    return [DLQEntry.from_json(r) for r in raw_entries]


async def get_dlq_size(redis: Any) -> int:
    """Return current DLQ length. Used by /health/ready and /metrics."""
    return int(await redis.llen(DLQ_KEY))


async def requeue_from_dlq(
    redis: Any,
    *,
    job_id: str,
    queue_client: QueueClient,
) -> str | None:
    """Pop the entry whose ``job_id`` matches and re-enqueue it.

    Returns the new job_id (which equals the original — we requeue with the
    same idempotency_key) or None if no DLQ entry matched.

    Linear scan over the DLQ list. Fine for bootcamp scale; production would
    keep an index.
    """
    raw_entries: list[bytes] = await redis.lrange(DLQ_KEY, 0, -1)
    for raw in raw_entries:
        entry = DLQEntry.from_json(raw)
        if entry.job_id == job_id:
            # LREM 1 raw — remove exactly this entry.
            await redis.lrem(DLQ_KEY, 1, raw)
            # Clear the DEAD_LETTER JobResult so the requeued run can write a
            # fresh one.
            await redis.delete(f"job_result:{job_id}")
            new_job_id = await queue_client.enqueue(
                entry.task_name,
                entry.payload,
                idempotency_key=job_id,
            )
            log.info(
                "dlq_requeued",
                old_job_id=job_id,
                new_job_id=new_job_id,
                task_name=entry.task_name,
            )
            return new_job_id
    return None


async def purge_dlq(redis: Any, *, older_than: timedelta | None = None) -> int:
    """Remove DLQ entries older than ``older_than``. Default: purge all.

    Returns count purged. Linear scan; fine for bootcamp scale.
    """
    raw_entries: list[bytes] = await redis.lrange(DLQ_KEY, 0, -1)
    if older_than is None:
        await redis.delete(DLQ_KEY)
        log.info("dlq_purged_all", count=len(raw_entries))
        return len(raw_entries)

    cutoff = datetime.now(UTC) - older_than
    purged = 0
    for raw in raw_entries:
        entry = DLQEntry.from_json(raw)
        try:
            enqueued = datetime.fromisoformat(entry.enqueued_at)
        except ValueError:
            continue
        if enqueued < cutoff:
            await redis.lrem(DLQ_KEY, 1, raw)
            purged += 1
    log.info("dlq_purged", count=purged, older_than_seconds=older_than.total_seconds())
    return purged
