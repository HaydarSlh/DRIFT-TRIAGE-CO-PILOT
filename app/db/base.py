"""SQLAlchemy 2.0 declarative base and async engine factory.

The engine and sessionmaker are created lazily by ``build_engine`` so the lifespan
hook in ``app.main`` can own their lifecycle and dispose cleanly on shutdown.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession as AsyncSession
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base. All ORM models in ``app.db.models`` inherit from this.

    Alembic's ``target_metadata`` reads ``Base.metadata`` to discover the schema
    for autogeneration; keep new models in the ``app.db.models`` import path so
    they're picked up.
    """


def build_engine(url: str) -> AsyncEngine:
    """Build an async SQLAlchemy engine.

    Args:
        url: SQLAlchemy URL with the asyncpg driver, e.g.
            ``postgresql+asyncpg://user:pass@host:5432/db``.
    """
    return create_async_engine(url, pool_pre_ping=True, future=True)


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build an ``async_sessionmaker`` bound to ``engine``.

    ``expire_on_commit=False`` is critical for FastAPI: after a commit we still
    want to read attributes off the persisted object inside the same request,
    and the default expiry behavior triggers a fresh SELECT on every access.
    """
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
