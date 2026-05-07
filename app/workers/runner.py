"""arq worker entry point.

What happens at boot:

1. Import ``app.workers`` (this triggers the side-effect imports of every
   task module, which registers them with ``TASK_REGISTRY``).
2. Build a ``WorkerSettings`` class on the fly from ``Settings``.
3. Hand off to ``arq.worker.run_worker``.

Graceful shutdown: arq drains in-flight jobs up to the worker's job timeout
and then exits. We don't override that — arq does the right thing and
fighting it just adds bugs.

Run as:
    uv run python -m app.workers.runner
or:
    uv run cp4-worker
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from urllib.parse import urlparse

from arq.connections import RedisSettings

# Trigger task module imports so TASK_REGISTRY is populated.
import app.workers  # noqa: F401  (intentional side-effect import)
from app.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.taskqueue.tasks import TASK_REGISTRY

log = get_logger(__name__)


def _redis_settings_from_url(url: str) -> RedisSettings:
    """Translate a ``redis://host:port/db`` URL to arq's ``RedisSettings``."""
    parsed = urlparse(url)
    db = int(parsed.path.lstrip("/")) if parsed.path else 0
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=db,
        password=parsed.password,
    )


async def _on_startup(ctx: dict[str, Any]) -> None:
    """Worker startup hook. arq passes ``ctx``; we stash a settings ref."""
    ctx["settings"] = get_settings()
    log.info("worker_started", concurrency=ctx["settings"].worker_concurrency)


async def _on_shutdown(ctx: dict[str, Any]) -> None:
    """Worker shutdown hook — for symmetry with startup."""
    log.info("worker_shutting_down")


def build_worker_settings(settings: Settings | None = None) -> type:
    """Construct an arq-compatible ``WorkerSettings`` class.

    The class form (rather than a config dict) is what arq expects in its
    CLI, but at runtime it just reads the attributes. Building it
    dynamically keeps ``Settings`` as the single source of truth.
    """
    from typing import ClassVar

    s: Settings = settings or get_settings()

    class WorkerSettings:
        functions: ClassVar = list(TASK_REGISTRY.values())
        redis_settings = _redis_settings_from_url(s.queue_redis_url)
        max_jobs = s.worker_concurrency
        job_timeout = s.job_timeout_s
        # arq's own retry defaults to 5; we keep it low because tenacity
        # inside @task already retries transient errors. arq-level retries
        # exist to handle worker crashes.
        max_tries = 2
        on_startup = staticmethod(_on_startup)
        on_shutdown = staticmethod(_on_shutdown)
        # Keep result keys around long enough for the `await_result` loop
        # plus a generous buffer.
        keep_result = s.job_timeout_s + 60

    WorkerSettings.__name__ = "DriftTriageWorkerSettings"
    return WorkerSettings


def main() -> None:
    """Console-script entry point. Boots arq via its programmatic API.

    arq has its own ``arq`` CLI; we shell out to ``arq.worker.run_worker``
    directly so the FastAPI logging configuration is applied first.
    """
    settings = get_settings()
    configure_logging(level=settings.log_level, env=settings.app_env)

    from arq.worker import run_worker

    worker_settings = build_worker_settings(settings)
    try:
        run_worker(worker_settings)
    except KeyboardInterrupt:
        log.info("worker_interrupted")
        sys.exit(0)


# Allow ``python -m app.workers.runner``.
if __name__ == "__main__":  # pragma: no cover
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    main()
