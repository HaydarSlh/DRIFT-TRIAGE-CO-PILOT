"""Queue admin CLI.

Commands:
    uv run python -m app.cli.queue_admin status
        — queue depth, worker count, DLQ size

    uv run python -m app.cli.queue_admin dlq list [--limit N]
        — show pending DLQ items

    uv run python -m app.cli.queue_admin dlq requeue <job_id>
        — re-enqueue a DLQ entry

    uv run python -m app.cli.queue_admin dlq purge [--older-than 7d]
        — purge old DLQ entries

For an in-cluster operator the API endpoints in ``app.api.jobs`` are the
preferred surface. The CLI is for offline poking — laptop debugging, or
when the API is down because of the very thing you're trying to fix.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import timedelta

from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.deps import build_queue_client
from app.taskqueue.dlq import (
    get_dlq_size,
    list_dlq,
    purge_dlq,
    requeue_from_dlq,
)

log = get_logger(__name__)


# ---------- helpers ---------------------------------------------------------


def _parse_duration(spec: str) -> timedelta:
    """Parse '7d', '12h', '30m', '90s' → timedelta. Bare number → seconds."""
    m = re.fullmatch(r"\s*(\d+)\s*([smhd]?)\s*", spec)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid duration: {spec!r}")
    n = int(m.group(1))
    unit = m.group(2) or "s"
    return {
        "s": timedelta(seconds=n),
        "m": timedelta(minutes=n),
        "h": timedelta(hours=n),
        "d": timedelta(days=n),
    }[unit]


# ---------- commands --------------------------------------------------------


async def _cmd_status(_args: argparse.Namespace) -> int:
    settings = get_settings()
    qc = await build_queue_client(settings)
    dlq_size = await get_dlq_size(qc._redis)
    workers_alive = bool(await qc._arq.exists("arq:health-check"))
    summary = {
        "queue_redis_url": settings.queue_redis_url,
        "workers_alive": workers_alive,
        "dlq_size": dlq_size,
        "configured_concurrency": settings.worker_concurrency,
    }
    print(json.dumps(summary, indent=2))
    return 0


async def _cmd_dlq_list(args: argparse.Namespace) -> int:
    qc = await build_queue_client()
    entries = await list_dlq(qc._redis, limit=args.limit)
    print(
        json.dumps(
            [
                {
                    "job_id": e.job_id,
                    "task_name": e.task_name,
                    "attempts": e.attempts,
                    "error": e.error,
                    "enqueued_at": e.enqueued_at,
                    "payload": e.payload,
                }
                for e in entries
            ],
            indent=2,
            default=str,
        )
    )
    return 0


async def _cmd_dlq_requeue(args: argparse.Namespace) -> int:
    qc = await build_queue_client()
    new_id = await requeue_from_dlq(qc._redis, job_id=args.job_id, queue_client=qc)
    if new_id is None:
        print(f"no DLQ entry for {args.job_id}", file=sys.stderr)
        return 2
    print(json.dumps({"old_job_id": args.job_id, "new_job_id": new_id}))
    return 0


async def _cmd_dlq_purge(args: argparse.Namespace) -> int:
    qc = await build_queue_client()
    older = _parse_duration(args.older_than) if args.older_than else None
    purged = await purge_dlq(qc._redis, older_than=older)
    print(json.dumps({"purged": purged}))
    return 0


# ---------- argparse wiring ------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cp4-queue-admin",
        description="Inspect and operate the CP4 queue + DLQ.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="queue depth, worker count, DLQ size")

    dlq = sub.add_parser("dlq", help="dead-letter queue commands")
    dlq_sub = dlq.add_subparsers(dest="dlq_cmd", required=True)

    dlq_list = dlq_sub.add_parser("list", help="show pending DLQ items")
    dlq_list.add_argument("--limit", type=int, default=100)

    dlq_req = dlq_sub.add_parser("requeue", help="re-enqueue a DLQ entry")
    dlq_req.add_argument("job_id")

    dlq_purge = dlq_sub.add_parser("purge", help="purge old DLQ entries")
    dlq_purge.add_argument(
        "--older-than",
        help="duration like 7d / 12h / 30m / 90s. Omit to purge everything.",
        default=None,
    )

    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.cmd == "status":
        return await _cmd_status(args)
    if args.cmd == "dlq":
        if args.dlq_cmd == "list":
            return await _cmd_dlq_list(args)
        if args.dlq_cmd == "requeue":
            return await _cmd_dlq_requeue(args)
        if args.dlq_cmd == "purge":
            return await _cmd_dlq_purge(args)
    return 1


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, env=settings.app_env)
    args = _build_parser().parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":  # pragma: no cover
    main()
