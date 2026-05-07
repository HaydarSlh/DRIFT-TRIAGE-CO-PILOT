"""Job result records and Redis-backed storage helpers.

A ``JobResult`` is the terminal state of a queued job. The worker persists one
to Redis when the job finishes (successfully OR after exhausting retries) and
the producer reads it back via ``QueueClient.await_result``.

We store JobResults under ``job_result:{job_id}`` with a configurable TTL so
the keyspace doesn't grow unbounded. The TTL is roughly "how long after a job
finishes might a producer still ask about it" — default an hour, override via
``QueueClient.enqueue(..., ttl=...)``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis

JOB_RESULT_KEY_PREFIX = "job_result:"


class JobStatus(StrEnum):
    """Lifecycle states a producer can observe for a job.

    The worker only ever writes ``completed`` or ``failed``. ``queued`` /
    ``in_progress`` are inferred — if no result key exists yet but the job is
    in arq's queue/in-flight set, status is queued or in_progress respectively.
    ``dead_letter`` is set explicitly by the DLQ writer.
    """

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class JobResult:
    """Terminal record produced when a job stops running.

    ``result_data`` holds the worker's return value when ``status`` is
    completed. ``error`` holds a stringified exception when failed. ``attempts``
    counts every actual function invocation (initial + tenacity retries +
    arq-level redeliveries).
    """

    job_id: str
    status: JobStatus
    result_data: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 1
    completed_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d, default=str)

    @classmethod
    def from_json(cls, raw: str) -> JobResult:
        d = json.loads(raw)
        d["status"] = JobStatus(d["status"])
        return cls(**d)


def _result_key(job_id: str) -> str:
    return f"{JOB_RESULT_KEY_PREFIX}{job_id}"


async def store_result(redis: Redis, result: JobResult, *, ttl: int = 3600) -> None:
    """Persist ``result`` under ``job_result:{job_id}`` with the given TTL."""
    await redis.set(_result_key(result.job_id), result.to_json(), ex=ttl)


async def load_result(redis: Redis, job_id: str) -> JobResult | None:
    """Read a previously stored ``JobResult`` or return None if not yet written.

    Returning None means "the worker has not finished" — a producer should
    interpret that as ``queued`` or ``in_progress`` (use the queue's own
    status check, not this one, to disambiguate).
    """
    raw = await redis.get(_result_key(job_id))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    return JobResult.from_json(raw)
