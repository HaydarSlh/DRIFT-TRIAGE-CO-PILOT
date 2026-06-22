"""LangGraph checkpointer factory.

The checkpointer is the *short-term* memory of the agent system. It stores per-step
graph state keyed by ``thread_id`` so a graph can be paused (e.g. on a human review
interrupt), the process can die, and the run can resume from the last good step
when the process comes back up.

Postgres holds the *long-term* domain memory (Reports, audit trail, reviews); Redis
holds the *short-term* graph memory. Mixing the two in one store is technically
possible but conflates concerns and makes it harder to expire short-term state.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


async def build_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    """Build and initialize a Redis-backed async checkpointer.

    Args:
        settings: Application settings; uses ``redis_url``.

    Returns:
        A live ``AsyncRedisSaver`` ready to be passed to ``graph.compile``.
    """
    saver = AsyncRedisSaver(redis_url=settings.redis_url)
    # asetup() creates the Redis indexes the saver needs; safe to call repeatedly.
    await saver.asetup()
    log.info("checkpointer_ready", url=settings.redis_url)
    return saver
