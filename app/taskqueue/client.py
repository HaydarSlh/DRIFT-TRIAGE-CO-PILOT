"""Producer-side queue API.

The agent only ever uses ``QueueClient``; it does not import arq directly.
Two reasons:

1. The queue can be swapped (arq → saq → Celery → in-memory test fake)
   without touching agent code.
2. Idempotency is enforced at the producer boundary, not inside the worker.
   By making ``enqueue`` deterministic when an ``idempotency_key`` is given,
   we get "submit twice, run once" with no help needed from the worker.

API:
- ``enqueue(task_name, payload, *, idempotency_key=None, ttl=3600) -> job_id``
- ``await_result(job_id, *, timeout=120) -> JobResult``
- ``get_status(job_id) -> JobStatus``
- ``cancel(job_id) -> bool``

The ``await_result`` polling loop is deliberate. arq has a result-pubsub
hookup we could use but it adds a Redis subscription per producer; for the
bootcamp the polling cost is invisible and the code is dramatically simpler.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from app.config import Settings
from app.core.errors import ToolError
from app.core.logging import get_logger
from app.taskqueue.results import JobResult, JobStatus, load_result

log = get_logger(__name__)


class QueueClient:
    """Wraps an arq pool with the simpler ``QueueClient`` API.

    Construction:
        - ``arq_pool``: an ``arq.ArqRedis`` (returned by ``arq.create_pool``).
          This is the producer-side connection used to enqueue jobs.
        - ``redis``: a ``redis.asyncio.Redis`` for direct reads of result
          keys / DLQ list. Same Redis instance, distinct logical roles.
        - ``settings``: where TTL/timeout/idempotency defaults come from.

    Tests can substitute either side. For the unit tests in
    ``tests/queue/`` we use ``fakeredis.aioredis.FakeRedis`` for ``redis``
    and a tiny in-memory stub for ``arq_pool`` — see
    ``app.testing.queue_fakes`` (added in CP4).
    """

    def __init__(
        self,
        *,
        arq_pool: Any,
        redis: Any,
        settings: Settings,
    ) -> None:
        self._arq = arq_pool
        self._redis = redis
        self._settings = settings

    # ------------------------------------------------------------------ #
    # Producer methods
    # ------------------------------------------------------------------ #

    async def enqueue(
        self,
        task_name: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        ttl: int = 3600,
    ) -> str:
        """Submit a job and return its id.

        When ``idempotency_key`` is provided the job_id is the SHA-256 of
        ``task_name|idempotency_key`` (truncated to 16 chars). arq will refuse
        a second enqueue with the same id while the first is queued/running,
        so the duplicate just re-uses the existing job. We additionally
        check whether a final ``JobResult`` already exists; if so, we skip
        the enqueue entirely and return the existing job_id — same effect
        ("submit twice, run once") but resilient even after the first run
        has finished.

        Args:
            task_name: Registered worker task name (must be in
                ``app.queue.tasks.TASK_REGISTRY``).
            payload: JSON-serializable dict the task will receive.
            idempotency_key: Optional. If set, makes the job_id deterministic.
            ttl: How long to keep the JobResult key alive after completion.

        Returns:
            The job_id (string).
        """
        job_id = (
            self._deterministic_job_id(task_name, idempotency_key)
            if idempotency_key
            else None
        )

        # Pre-check: did we already finish a job under this idempotency key?
        if job_id is not None:
            existing = await load_result(self._redis, job_id)
            if existing is not None:
                log.info(
                    "enqueue_idempotent_skip",
                    task_name=task_name,
                    job_id=job_id,
                    status=existing.status.value,
                )
                return job_id

        # arq's enqueue_job returns ``None`` if a job with this _job_id is
        # already queued/in-flight. That's fine — duplicate enqueue, single
        # execution. Same job_id either way.
        enqueue_kwargs: dict[str, Any] = {"_expires": ttl}
        if job_id is not None:
            enqueue_kwargs["_job_id"] = job_id
        job = await self._arq.enqueue_job(task_name, payload, **enqueue_kwargs)

        # If arq returns a Job object, take its id; else use our computed one.
        # (Tests against the in-memory stub return our preset id directly.)
        if job is not None and getattr(job, "job_id", None):
            return str(job.job_id)
        if job_id is not None:
            return job_id
        # arq generated the id on its own (no idempotency_key path) but
        # somehow returned None. Treat as transient queue error.
        raise ToolError(f"queue rejected job '{task_name}' without returning an id")

    async def await_result(
        self,
        job_id: str,
        *,
        timeout: float = 120.0,
        poll_interval: float = 0.25,
    ) -> JobResult:
        """Block until the job finishes (or ``timeout`` expires).

        Polls ``job_result:{job_id}``. The polling cadence is intentionally
        coarse — for tools that take 5-30 seconds the extra latency is in
        the noise, and a tighter loop turns into Redis chatter.

        Raises:
            TimeoutError: If no terminal result appears within ``timeout``.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            result = await load_result(self._redis, job_id)
            if result is not None and result.status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.DEAD_LETTER,
            }:
                return result
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"job {job_id} did not complete within {timeout:.0f}s"
                )
            await asyncio.sleep(poll_interval)

    async def get_status(self, job_id: str) -> JobStatus:
        """Best-effort status lookup.

        Order of precedence:
            1. Stored JobResult → its terminal status.
            2. arq says the job is in-flight → IN_PROGRESS.
            3. arq says the job is queued → QUEUED.
            4. Fallback → QUEUED (we don't know, but the producer just
               enqueued so this is the safest default).
        """
        result = await load_result(self._redis, job_id)
        if result is not None:
            return result.status

        # arq stores live jobs under arq:in_progress:{job_id} — but its API
        # surface for "is this job alive" is small. We do a best-effort key
        # check that doesn't break tests that bypass arq entirely.
        try:
            in_progress = await self._redis.exists(f"arq:in-progress:{job_id}")
            if in_progress:
                return JobStatus.IN_PROGRESS
        except Exception:
            pass
        return JobStatus.QUEUED

    async def cancel(self, job_id: str) -> bool:
        """Best-effort cancellation. Returns True if arq removed the job.

        Already-running jobs cannot be cancelled mid-execution from the
        producer side — the worker honors ``JOB_TIMEOUT_S`` for that.
        """
        try:
            return bool(await self._arq.abort_job(job_id))
        except Exception as exc:
            log.warning("cancel_failed", job_id=job_id, error=str(exc))
            return False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deterministic_job_id(task_name: str, key: str) -> str:
        """SHA-256 of ``task_name|key`` truncated to 16 hex chars.

        Truncation is a deliberate space/collision tradeoff. 16 chars = 64
        bits = ~4 billion job ids before a collision becomes likely. For
        bootcamp scale, that is more than fine.
        """
        h = hashlib.sha256(f"{task_name}|{key}".encode()).hexdigest()
        return h[:16]
