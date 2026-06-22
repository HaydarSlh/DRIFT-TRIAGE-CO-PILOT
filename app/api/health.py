"""Health and readiness endpoints."""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import HumanMessage
from sqlalchemy import text

from app.agents import llm as llm_module
from app.config import Settings
from app.core.logging import get_logger
from app.deps import get_settings_dep
from app.taskqueue.dlq import get_dlq_size

router = APIRouter(prefix="/health", tags=["health"])
log = get_logger(__name__)

SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


@router.get("")
async def health() -> dict[str, str]:
    """Liveness probe.

    Returns ``{"status": "ok"}`` whenever the process is running. Does NOT touch
    the LLM provider — orchestrators that call this on a tight loop must not be
    rate-limited by upstream APIs.
    """
    return {"status": "ok"}


@router.get("/llm")
async def health_llm(settings: SettingsDep) -> dict[str, Any]:
    """Readiness probe for the configured LLM provider.

    Performs a tiny ping and reports the round-trip latency. Failures return
    ``status="fail"`` with HTTP 200 so orchestrators can read the body without
    triggering retry storms.
    """
    start = time.perf_counter()
    try:
        chat = llm_module.get_chat_model("classifier", settings=settings)
        if settings.llm_provider == "google":
            chat = chat.bind(max_output_tokens=1)
        else:
            chat = chat.bind(max_tokens=1)
        await chat.ainvoke([HumanMessage(content="ping")])
    except Exception as exc:
        log.warning("health_llm_fail", error=str(exc))
        return {"status": "fail", "provider": settings.llm_provider, "error": str(exc)}
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {"status": "ok", "provider": settings.llm_provider, "latency_ms": latency_ms}


@router.get("/ready")
async def health_ready(request: Request, settings: SettingsDep) -> dict[str, Any]:
    """Readiness probe.

    Distinguishes "app failed" (cannot serve any request) from "app degraded"
    (can serve some requests but a queue worker is down). Postgres or Redis
    failure → failed. Workers down → degraded.

    Both responses are HTTP 200; the body's ``status`` field carries the
    actual signal. A real K8s probe would inspect the body or have a
    second endpoint that returns 503 for failed — for the bootcamp the
    body is enough and tests are simpler.
    """
    checks: dict[str, Any] = {}

    try:
        sessionmaker = request.app.state.sessionmaker
        async with sessionmaker() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        log.warning("ready_postgres_fail", error=str(exc))
        checks["postgres"] = f"fail: {exc}"
        return {"status": "failed", "checks": checks}

    try:
        saver = request.app.state.checkpointer
        client = getattr(saver, "_redis_client", None) or getattr(saver, "redis", None)
        if client is not None and hasattr(client, "ping"):
            await client.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        log.warning("ready_redis_fail", error=str(exc))
        checks["redis"] = f"fail: {exc}"
        return {"status": "failed", "checks": checks}

    queue = getattr(request.app.state, "queue_client", None)
    if queue is None:
        checks["queue"] = "disabled"
    else:
        try:
            workers_alive = await queue._arq.exists("arq:queue:health-check")
            dlq_depth = await get_dlq_size(queue._redis)
            checks["queue"] = "ok"
            checks["workers"] = "ok" if workers_alive else "no_workers_responding"
            checks["dlq_size"] = dlq_depth
        except Exception as exc:
            log.warning("ready_queue_fail", error=str(exc))
            checks["queue"] = f"degraded: {exc}"

    degraded = (
        checks.get("workers") == "no_workers_responding"
        or "degraded" in str(checks.get("queue", ""))
    )
    return {"status": "degraded" if degraded else "ok", "checks": checks}


@router.get("/metrics")
async def metrics(request: Request, settings: SettingsDep) -> dict[str, Any]:
    """Tiny metrics surface for the bootcamp.

    Real deployments expose Prometheus; we return hand-rolled JSON with
    queue depth, DLQ depth, and worker pool size. Production observability
    lives in Week 7.
    """
    queue = getattr(request.app.state, "queue_client", None)
    payload: dict[str, Any] = {
        "worker_pool": {"configured_concurrency": settings.worker_concurrency},
    }
    if queue is not None:
        payload["dlq_size"] = await get_dlq_size(queue._redis)
    return payload
