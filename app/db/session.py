"""Per-request DB session dependency."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped ``AsyncSession``.

    Reads the lifespan-built ``async_sessionmaker`` from ``app.state.sessionmaker``
    and opens a session that's closed automatically when the request ends.

    Background tasks must NOT capture this session — by the time they run, the
    request is done and the session is closed. They should call
    ``request.app.state.sessionmaker()`` directly to open a fresh one.

    Yields:
        A new ``AsyncSession`` for the duration of the request.
    """
    sessionmaker = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session
