"""FastAPI application factory.

Lifespan wires up:
    - a SQLAlchemy ``AsyncEngine`` + ``async_sessionmaker`` (Postgres)
    - a Redis-backed LangGraph checkpointer
    - a ``QueueClient`` (arq pool + redis-asyncio) when ``ENABLE_QUEUE=true``

The queue client lives on ``app.state.queue_client``. The deps helper
``get_queue_client_dep`` reads it. When the queue is disabled, the
attribute is set to ``None`` so the app still serves health and
non-queue endpoints.
"""

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents.checkpointer import build_checkpointer
from app.agents.graph import build_graph
from app.api import approvals, health, investigations, jobs, webhooks
from app.config import get_settings
from app.core.errors import install_exception_handlers
from app.core.logging import configure_logging, get_logger, request_id_ctx
from app.db.base import build_engine, build_sessionmaker
from app.deps import build_queue_client
from app.taskqueue.client import QueueClient


def create_app(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    sessionmaker: async_sessionmaker | None = None,
    queue_client: QueueClient | None = None,
) -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    configure_logging(level=settings.log_level, env=settings.app_env)
    log = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        engine = None
        if sessionmaker is None:
            engine = build_engine(settings.database_url)
            app.state.sessionmaker = build_sessionmaker(engine)
            app.state.engine = engine
        else:
            app.state.sessionmaker = sessionmaker
            app.state.engine = None

        if checkpointer is None:
            app.state.checkpointer = await build_checkpointer(settings)
        else:
            app.state.checkpointer = checkpointer

        app.state.settings = settings

        if queue_client is not None:
            app.state.queue_client = queue_client
        elif settings.enable_queue:
            try:
                app.state.queue_client = await build_queue_client(settings)
            except Exception as exc:
                log.warning("queue_init_failed", error=str(exc))
                app.state.queue_client = None
        else:
            app.state.queue_client = None

        app.state.graph = build_graph(checkpointer=app.state.checkpointer)
        log.info(
            "app_ready",
            env=settings.app_env,
            provider=settings.llm_provider,
            queue=app.state.queue_client is not None,
        )
        try:
            yield
        finally:
            if engine is not None:
                await engine.dispose()
            saver = app.state.checkpointer
            close = getattr(saver, "aclose", None)
            if callable(close):
                await close()
            qc = getattr(app.state, "queue_client", None)
            if qc is not None:
                qc_close = getattr(qc._arq, "close", None)
                if callable(qc_close):
                    await qc_close()

    app = FastAPI(title="drift-triage-agent", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid4().hex
        token = request_id_ctx.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["x-request-id"] = rid
        return response

    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(investigations.router)
    app.include_router(approvals.router)
    app.include_router(jobs.router)
    app.include_router(webhooks.router)
    return app


app = create_app()