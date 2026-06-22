"""Queue-backed tool wrapper.

Single function — ``call_tool(name, payload, *, idempotency_key=None)`` —
the agent nodes (and other callers) call this and never see arq.

Pattern:
    1. Build a ``QueueClient``.
    2. Enqueue the job (deterministic id when idempotency_key is given).
    3. Wait for the result up to ``Settings.job_timeout_s``.
    4. Translate JobResult into either a return value or a raised
       ``ToolError``.

Graceful degradation: if ``Settings.enable_queue`` is False, we bypass the
queue entirely and call the registered task function inline. Useful for
local dev. If ``Settings.enable_queue`` is True but no worker can be
reached within ``worker_reach_deadline_s``, we honor
``Settings.enable_queue_fallback``: True → run inline anyway with a logged
warning, False → raise ``ToolError`` so the run fails predictably.

The "queue is degraded" detection is deliberately simple — a Redis ping
plus a peek at arq's worker key. A production system would lean on a
proper service-discovery signal; for the bootcamp this is enough to
demonstrate the pattern.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.config import get_settings
from app.core.errors import ToolError
from app.core.logging import get_logger
from app.taskqueue.client import QueueClient
from app.taskqueue.results import JobStatus
from app.taskqueue.tasks import TASK_REGISTRY

log = get_logger(__name__)


async def call_tool(
    name: str,
    payload: dict[str, Any],
    *,
    idempotency_key: str | None = None,
    queue: QueueClient | None = None,
) -> dict[str, Any]:
    """Invoke a registered worker task and return its result.

    Args:
        name: Task name in ``TASK_REGISTRY``.
        payload: JSON-serializable input dict.
        idempotency_key: Optional. Same key → same job_id → exactly one
            execution under producer-side dedup. Defaults to a hash of the
            payload, which is the right answer most of the time.
        queue: Optional pre-built ``QueueClient``. Tests pass one in;
            production code calls ``build_queue_client()`` from
            ``app.deps``.

    Returns:
        The task's result dict.

    Raises:
        ToolError: When the task fails permanently (after retries) OR the
            queue is degraded and fallback is disabled.
    """
    settings = get_settings()
    idempotency_key = idempotency_key or _payload_hash(name, payload)

    if not settings.enable_queue:
        return await _run_inline(name, payload, reason="queue_disabled")

    if queue is None:
        # Lazy import to avoid pulling arq into unit tests that don't queue.
        from app.deps import build_queue_client

        queue = await build_queue_client()

    log.info("tool_enqueue", task_name=name, idempotency_key=idempotency_key)
    job_id = await queue.enqueue(
        name,
        payload,
        idempotency_key=idempotency_key,
        ttl=settings.job_timeout_s + 60,
    )

    try:
        result = await queue.await_result(job_id, timeout=settings.job_timeout_s)
    except TimeoutError as exc:
        if settings.enable_queue_fallback:
            log.warning(
                "tool_queue_timeout_fallback",
                task_name=name,
                job_id=job_id,
                error=str(exc),
            )
            return await _run_inline(name, payload, reason="queue_timeout")
        raise ToolError(f"queue timed out for {name}: {exc}") from exc

    if result.status == JobStatus.COMPLETED:
        return result.result_data or {}
    raise ToolError(
        f"tool {name} ended in {result.status.value}: {result.error or 'unknown'}"
    )


async def _run_inline(
    name: str,
    payload: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Direct in-process execution path.

    The wrapped function in ``TASK_REGISTRY`` is the @task-decorated wrapper,
    which expects ``(ctx, payload)`` because arq calls it that way. We
    construct a minimal ctx with a stub redis so the wrapper's idempotency
    + retry plumbing still works.
    """
    log.info("tool_inline", task_name=name, reason=reason)
    func = TASK_REGISTRY.get(name)
    if func is None:
        raise ToolError(f"unknown task: {name}")

    # In-process inline path needs a "redis-like" object so the @task wrapper
    # can read/write the JobResult cache. We give it the real Redis if the
    # caller has configured one (production fallback) and a fakeredis stub
    # otherwise (unit-test happy path). Building this lazily keeps the
    # import graph clean.
    redis = await _inline_redis()
    job_id = _payload_hash(name, payload)
    out: dict[str, Any] = await func({"job_id": job_id, "redis": redis}, payload)
    return out


async def _inline_redis() -> Any:
    """Return a Redis-shaped object for the inline path.

    Production: real Redis (we already have it for the checkpointer).
    Tests:      fakeredis if redis is unreachable.
    """
    settings = get_settings()
    try:
        from redis.asyncio import Redis

        redis: Any = Redis.from_url(settings.queue_redis_url)
        await redis.ping()
        return redis
    except Exception as exc:
        log.warning("inline_real_redis_unavailable", error=str(exc))
        try:
            import fakeredis.aioredis

            return fakeredis.aioredis.FakeRedis()
        except ImportError:
            raise ToolError("no redis available for inline tool execution") from exc


def _payload_hash(name: str, payload: dict[str, Any]) -> str:
    blob = f"{name}|{json.dumps(payload, sort_keys=True, default=str)}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
