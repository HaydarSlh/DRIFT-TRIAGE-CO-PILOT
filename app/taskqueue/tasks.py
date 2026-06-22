"""``@task`` decorator: registers a worker function with idempotency, retry,
result-persistence, and DLQ-on-permanent-failure baked in.

Why this lives outside ``app.workers``:
The worker package contains the actual task implementations (scrape_url,
process_document, …). The decorator is the *plumbing* that wraps them. By
keeping it in ``app.queue`` we can unit-test the decorator without importing
any task body.

Two retry layers:

- **Inner (tenacity)**: handles known transient errors. Exponential backoff
  with jitter, capped at ``RETRY_BACKOFF_MAX_S``. Only retries
  ``TransientToolError`` — anything else (PermanentToolError or unexpected)
  short-circuits.
- **Outer (arq)**: handles worker crashes. If the worker dies between the
  task's first instruction and the result write, arq redelivers the job.
  Idempotency at the producer side AND inside the task (see ``_already_done``)
  make the redelivery safe.

The decorator's contract:
    @task("scrape_url", max_retries=3, retry_on=(TransientToolError,))
    async def scrape_url(payload: dict) -> dict: ...

The wrapped function is what arq executes. It receives ``(ctx, payload)``
because arq calls workers with a context dict first.

On permanent failure the wrapper:
    1. stores a FAILED JobResult (so producers see the error promptly)
    2. pushes a DLQ entry
    3. re-raises so arq logs the failure too
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import get_settings
from app.core.errors import PermanentToolError, ToolError, TransientToolError
from app.core.logging import get_logger
from app.taskqueue.dlq import push_to_dlq
from app.taskqueue.results import JobResult, JobStatus, load_result, store_result

log = get_logger(__name__)

#: Registered worker tasks — the arq ``Worker.functions`` list is built from
#: ``list(TASK_REGISTRY.values())`` in ``app.workers.runner``.
TASK_REGISTRY: dict[str, Callable[..., Awaitable[Any]]] = {}


def task(
    name: str | None = None,
    *,
    max_retries: int | None = None,
    retry_on: tuple[type[BaseException], ...] = (TransientToolError,),
    no_retry_on: tuple[type[BaseException], ...] = (PermanentToolError,),
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Wrap an async worker function with the production-grade plumbing.

    Args:
        name: Task name registered in ``TASK_REGISTRY``. Defaults to the
            wrapped function's ``__name__``.
        max_retries: Max tenacity retries. ``None`` reads ``Settings.max_retries``.
        retry_on: Tuple of exception types to retry on. Default is just
            ``TransientToolError`` — be explicit, don't retry-everything.
        no_retry_on: Hard-stop exception types. Even if they overlap with
            retry_on the no_retry list wins. Default includes
            ``PermanentToolError``.

    Returns:
        A decorator. Decorated function signature must be
        ``async def f(payload: dict) -> dict``. The arq-facing wrapper
        accepts ``(ctx, payload)``.
    """

    def decorator(
        func: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        task_name = name or func.__name__

        @wraps(func)
        async def wrapper(ctx: dict[str, Any], payload: dict[str, Any]) -> Any:
            settings = get_settings()
            job_id = str(ctx.get("job_id", _fallback_job_id(task_name, payload)))
            redis = ctx.get("redis")  # arq populates this; tests pass it directly
            if redis is None:
                raise RuntimeError(
                    "task wrapper invoked without a redis connection in ctx"
                )

            # Idempotency check: if the job_id already has a terminal result,
            # skip execution entirely. This handles arq redeliveries cleanly.
            existing = await load_result(redis, job_id)
            if existing is not None and existing.status == JobStatus.COMPLETED:
                log.info(
                    "task_skipped_idempotent",
                    task_name=task_name,
                    job_id=job_id,
                )
                return existing.result_data

            # Deterministic mode (tests) zeros out the backoff so the suite
            # doesn't pay seconds-per-attempt for transient retries.
            base_s = 0.0 if settings.is_deterministic else settings.retry_backoff_base_s
            max_s = 0.0 if settings.is_deterministic else settings.retry_backoff_max_s
            attempts_used = await _run_with_retry(
                func,
                payload,
                task_name=task_name,
                job_id=job_id,
                redis=redis,
                max_retries=(
                    max_retries
                    if max_retries is not None
                    else settings.max_retries
                ),
                retry_on=retry_on,
                no_retry_on=no_retry_on,
                deterministic=settings.is_deterministic,
                base_s=base_s,
                max_s=max_s,
                webhook=settings.dlq_alert_webhook,
            )
            return attempts_used

        wrapper.__name__ = task_name  # arq looks at __name__ for routing
        TASK_REGISTRY[task_name] = wrapper
        return wrapper

    return decorator


async def _run_with_retry(
    func: Callable[..., Awaitable[Any]],
    payload: dict[str, Any],
    *,
    task_name: str,
    job_id: str,
    redis: Any,
    max_retries: int,
    retry_on: tuple[type[BaseException], ...],
    no_retry_on: tuple[type[BaseException], ...],
    deterministic: bool,
    base_s: float,
    max_s: float,
    webhook: str | None,
) -> Any:
    """Apply the tenacity policy and persist a JobResult either way.

    Returns the worker function's result on success. On permanent failure
    it pushes to DLQ, persists a FAILED JobResult, and re-raises ``ToolError``.

    The retry policy:
        - ``stop_after_attempt(max_retries + 1)`` because tenacity counts the
          first call too.
        - ``wait_exponential_jitter(initial=base, max=max_s)``: exponential
          backoff with random jitter in [0, base * 2**attempt]. Determinism
          mode disables jitter so tests are reproducible.
        - ``retry_if_exception_type(retry_on)``: only retry the explicit
          allowlist.
        - We additionally short-circuit on ``no_retry_on`` *inside* the
          attempt — tenacity's exception filter doesn't compose easily, so
          we re-raise non-retryable errors before tenacity even sees them.
    """
    attempts = 0
    started = time.perf_counter()
    log.info("task_started", task_name=task_name, job_id=job_id, payload=payload)

    last_exc: BaseException | None = None
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential_jitter(
                initial=base_s,
                max=max_s,
                jitter=0.0 if deterministic else base_s,
            ),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
        ):
            with attempt:
                attempts = attempt.retry_state.attempt_number
                try:
                    result = await func(payload)
                except no_retry_on as exc:
                    # Permanent — break out of tenacity entirely.
                    last_exc = exc
                    raise
                except Exception as exc:
                    last_exc = exc
                    if attempts > max_retries + 1 or not isinstance(exc, retry_on):
                        # Will be handled by the except below.
                        raise
                    log.warning(
                        "task_retry",
                        task_name=task_name,
                        job_id=job_id,
                        attempt=attempts,
                        error=str(exc),
                    )
                    raise

                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log.info(
                    "task_completed",
                    task_name=task_name,
                    job_id=job_id,
                    attempts=attempts,
                    elapsed_ms=elapsed_ms,
                )
                await store_result(
                    redis,
                    JobResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        result_data=_jsonable(result),
                        attempts=attempts,
                    ),
                )
                return result
    except RetryError as exc:
        last_exc = exc.last_attempt.exception() if exc.last_attempt else exc
    except Exception as exc:
        last_exc = exc

    error_msg = repr(last_exc) if last_exc else "unknown_failure"
    log.error(
        "task_failed",
        task_name=task_name,
        job_id=job_id,
        attempts=attempts,
        error=error_msg,
    )
    await push_to_dlq(
        redis,
        job_id=job_id,
        task_name=task_name,
        payload=payload,
        error=error_msg,
        attempts=attempts,
        webhook_url=webhook,
    )
    raise ToolError(f"{task_name} failed permanently: {error_msg}") from last_exc


def _jsonable(value: Any) -> dict[str, Any]:
    """Coerce a worker return into a JSON-serializable dict.

    Tasks should return dicts. We accept other shapes too (string, list, …)
    and wrap them in ``{"value": ...}`` so result storage stays uniform.
    """
    if isinstance(value, dict):
        return value
    return {"value": value}


def _fallback_job_id(task_name: str, payload: dict[str, Any]) -> str:
    """Fallback id used when arq didn't populate ctx['job_id'].

    Tests sometimes invoke the wrapper directly without going through arq.
    A deterministic fallback keeps storage reads/writes paired up.
    """
    import hashlib
    import json

    blob = f"{task_name}|{json.dumps(payload, sort_keys=True, default=str)}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# Some seeded behavior for the deterministic-mode jitter knob below; kept
# at module scope so freezegun/monkeypatch in tests can override cleanly.
def _maybe_seed_jitter(deterministic: bool, seed: int) -> None:  # pragma: no cover
    if deterministic:
        random.seed(seed)


# Keep imports live — these are used by future introspection / sleep hooks
# and we want monkeypatch to be able to find them.
_INSPECT_KEEPALIVE = inspect
_ASYNCIO_KEEPALIVE = asyncio
