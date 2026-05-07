"""Pytest fixtures students can import in their tests.

The fixtures in this module are NOT auto-collected — they live here so test
files can `from app.testing.fixtures import patch_chat_model, clean_db, ...`
deliberately. The repo's per-test fixture wiring lives in ``tests/conftest.py``
and re-exports the ones it wants to be available everywhere.
"""

from collections.abc import AsyncIterator, Callable, Generator

import pytest
import pytest_asyncio
import redis.asyncio as redis_async
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.agents import llm as llm_module
from app.agents.nodes import _llm as legacy_llm
from app.testing.fakes import FakeChatModel
from app.testing.recorders import TrajectoryRecorder

# ---------------------------------------------------------------------------
# Chat-model patching
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[FakeChatModel], FakeChatModel]:
    """Return a function that installs a ``FakeChatModel`` in place of the real one.

    Use:

        def test_something(patch_chat_model):
            fake = patch_chat_model(FakeChatModel(...))
            ...

    Patches BOTH ``app.agents.llm.get_chat_model`` (CP3's canonical getter)
    AND ``app.agents.nodes._llm.get_llm`` (the back-compat shim) so legacy
    tests keep working during the migration.
    """

    def _install(fake: FakeChatModel) -> FakeChatModel:
        monkeypatch.setattr(
            llm_module, "get_chat_model", lambda _role, **_kwargs: fake
        )
        monkeypatch.setattr(legacy_llm, "get_llm", lambda _settings, **_kwargs: fake)
        return fake

    return _install


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def deterministic_mode(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Force-on the deterministic switch for every test.

    Auto-used so students don't have to remember to enable it. Production
    code is unaffected — this fixture only runs under pytest.
    """
    monkeypatch.setenv("DETERMINISTIC_MODE", "true")
    monkeypatch.setenv("TEST_DETERMINISTIC", "true")
    yield


# ---------------------------------------------------------------------------
# DB / Redis cleanup
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def clean_db(test_db_engine: AsyncEngine) -> AsyncIterator[AsyncEngine]:
    """Yield the test engine, truncate every table afterwards.

    Depends on the session-scoped ``test_db_engine`` fixture defined in
    ``tests/conftest.py``. If Postgres is unreachable, that fixture skips
    the test, and this fixture skips with it.
    """
    yield test_db_engine
    from app.db.base import Base

    async with test_db_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))


@pytest_asyncio.fixture
async def clean_redis() -> AsyncIterator[redis_async.Redis]:
    """Connect to Redis db=15, FLUSHDB before and after the test.

    Skips when Redis isn't reachable so unit tests run fine without
    infrastructure.
    """
    client = redis_async.from_url("redis://localhost:6379/15")
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis not reachable at localhost:6379; skipping Redis-bound tests")
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


# ---------------------------------------------------------------------------
# Trajectory recorder
# ---------------------------------------------------------------------------


@pytest.fixture
def trajectory_recorder() -> TrajectoryRecorder:
    """A fresh TrajectoryRecorder per test."""
    return TrajectoryRecorder()


# ---------------------------------------------------------------------------
# Session-style sessionmaker for tests that need their own
# ---------------------------------------------------------------------------


@pytest.fixture
def make_sessionmaker(
    test_sessionmaker: async_sessionmaker,
) -> async_sessionmaker:
    """Re-export of ``test_sessionmaker`` under a name that reads better in
    tests that use it standalone.
    """
    return test_sessionmaker


__all__ = [
    "clean_db",
    "clean_redis",
    "deterministic_mode",
    "make_sessionmaker",
    "patch_chat_model",
    "trajectory_recorder",
]
