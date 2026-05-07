"""Offline replay — reads from the local Redis checkpointer.

Usage:
    uv run python -m app.cli.replay_local <thread_id>

Useful when students don't have a LangSmith account but want to inspect a run
they ran themselves. Reads the LangGraph checkpointer for ``thread_id`` and
prints each checkpoint's state in time order.

Unlike ``replay.py``, this CLI does NOT scaffold a regression test — there's
no LLM-call history in a checkpoint, only state snapshots. For that, use the
LangSmith path. This is for "what state did my agent reach?" debugging.
"""

import argparse
import asyncio
import json
import sys
from typing import Any

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from pydantic import BaseModel

from app.config import get_settings
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


def _normalize(value: Any) -> Any:
    """Make checkpointer values JSON-serializable for printing."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


async def replay_local(thread_id: str) -> int:
    settings = get_settings()
    saver = AsyncRedisSaver(redis_url=settings.redis_url)
    await saver.asetup()

    config = {"configurable": {"thread_id": thread_id}}
    snapshots = []
    async for cp in saver.alist(config):
        snapshots.append(cp)
    if not snapshots:
        log.error("no_checkpoints_found", thread_id=thread_id)
        return 2

    # alist returns newest-first; flip so the print walks the run forward.
    snapshots.reverse()
    for i, cp in enumerate(snapshots):
        values = getattr(cp, "checkpoint", {}).get("channel_values", {}) or {}
        print(f"--- checkpoint {i + 1} / {len(snapshots)} ---")
        print(json.dumps(_normalize(values), indent=2, default=str))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cp3-replay-local",
        description="Replay a thread from the local Redis checkpointer.",
    )
    parser.add_argument("thread_id", help="The thread_id to replay.")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level=settings.log_level, env=settings.app_env)

    sys.exit(asyncio.run(replay_local(args.thread_id)))


if __name__ == "__main__":
    main()
