"""FastAPI dependency providers."""

from typing import Any
from urllib.parse import urlparse

from fastapi import Request

from app.config import Settings, get_settings
from app.taskqueue.client import QueueClient
from app.services.investigation_service import InvestigationService
from app.services.approval_service import ApprovalService


def get_settings_dep() -> Settings:
    """FastAPI dependency wrapping ``get_settings()``."""
    return get_settings()


def get_investigation_service(request: Request) -> InvestigationService:
    """Construct an ``InvestigationService`` from the lifespan-built sessionmaker and graph."""
    return InvestigationService(
        sessionmaker=request.app.state.sessionmaker,
        graph=request.app.state.graph,
    )


def get_approval_service(request: Request) -> ApprovalService:
    """Construct an ``ApprovalService`` that delegates resume to ``InvestigationService``."""
    investigation_service = get_investigation_service(request)
    return ApprovalService(
        sessionmaker=request.app.state.sessionmaker,
        investigation_service=investigation_service,
    )


async def build_queue_client(settings: Settings | None = None) -> QueueClient:
    """Construct a ``QueueClient`` backed by arq + a redis-asyncio connection."""

    settings = settings or get_settings()

    from arq import create_pool
    from arq.connections import RedisSettings
    from redis.asyncio import Redis

    parsed = urlparse(settings.queue_redis_url)
    db = int(parsed.path.lstrip("/")) if parsed.path else 0
    arq_settings = RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=db,
        password=parsed.password,
    )
    arq_pool: Any = await create_pool(arq_settings)
    redis_client: Any = Redis.from_url(settings.queue_redis_url)
    return QueueClient(arq_pool=arq_pool, redis=redis_client, settings=settings)


def get_queue_client_dep(request: Request) -> QueueClient | None:
    """FastAPI dependency: return the queue client stashed on app.state.

    May be None in tests that don't bring up the queue layer.
    """
    return getattr(request.app.state, "queue_client", None)