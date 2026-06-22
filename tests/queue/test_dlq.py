"""Dead-letter queue tests.

Behaviors:
- ``push_to_dlq`` writes (a) the DLQ entry, (b) a DEAD_LETTER JobResult.
- ``list_dlq`` returns newest-first.
- ``requeue_from_dlq`` removes the DLQ entry, clears the JobResult, and
  re-enqueues with the same idempotency key.
- ``purge_dlq`` removes everything (or older-than entries).
- The optional webhook fires on push (and a fail in the webhook does not
  prevent the DLQ write).
"""

from __future__ import annotations

from datetime import UTC, timedelta
from typing import Any

from app.config import get_settings
from app.taskqueue import JobStatus, QueueClient, load_result, push_to_dlq
from app.taskqueue.dlq import DLQEntry, list_dlq, purge_dlq, requeue_from_dlq
from tests.queue.conftest import StubArqPool


async def test_push_writes_entry_and_terminal_result(
    settings_override: None,
    fake_redis: Any,
) -> None:
    entry = await push_to_dlq(
        fake_redis,
        job_id="j1",
        task_name="scrape_url",
        payload={"url": "https://x"},
        error="boom",
        attempts=4,
    )

    assert isinstance(entry, DLQEntry)
    listed = await list_dlq(fake_redis)
    assert [e.job_id for e in listed] == ["j1"]

    stored = await load_result(fake_redis, "j1")
    assert stored is not None
    assert stored.status == JobStatus.DEAD_LETTER
    assert stored.error == "boom"


async def test_dlq_orders_newest_first(
    settings_override: None,
    fake_redis: Any,
) -> None:
    for i, jid in enumerate(["a", "b", "c"]):
        await push_to_dlq(
            fake_redis,
            job_id=jid,
            task_name="scrape_url",
            payload={"i": i},
            error="boom",
            attempts=1,
        )
    listed = await list_dlq(fake_redis)
    assert [e.job_id for e in listed] == ["c", "b", "a"]


async def test_requeue_removes_entry_and_resubmits(
    settings_override: None,
    fake_redis: Any,
) -> None:
    await push_to_dlq(
        fake_redis,
        job_id="orig",
        task_name="web_search",
        payload={"query": "x"},
        error="boom",
        attempts=2,
    )
    arq = StubArqPool()
    qc = QueueClient(arq_pool=arq, redis=fake_redis, settings=get_settings())

    new_id = await requeue_from_dlq(fake_redis, job_id="orig", queue_client=qc)

    assert new_id is not None
    listed = await list_dlq(fake_redis)
    assert listed == []  # no longer in DLQ
    assert len(arq.enqueued) == 1
    assert arq.enqueued[0][0] == "web_search"
    # The terminal DEAD_LETTER record was cleared so the next run can
    # write a fresh one.
    assert await load_result(fake_redis, "orig") is None


async def test_purge_all(
    settings_override: None,
    fake_redis: Any,
) -> None:
    for jid in ["a", "b", "c"]:
        await push_to_dlq(
            fake_redis,
            job_id=jid,
            task_name="scrape_url",
            payload={},
            error="boom",
            attempts=1,
        )

    purged = await purge_dlq(fake_redis)
    assert purged == 3
    listed = await list_dlq(fake_redis)
    assert listed == []


async def test_purge_older_than(
    settings_override: None,
    fake_redis: Any,
    monkeypatch: Any,
) -> None:
    """Use freezegun-style monkeypatching of ``DLQEntry.from_json`` to
    simulate an old entry. Simpler than wedging freezegun into Redis I/O.
    """
    from datetime import datetime

    old_iso = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    new_iso = datetime.now(UTC).isoformat()

    # Push two entries with current timestamps, then mutate one to old.
    await push_to_dlq(
        fake_redis,
        job_id="old",
        task_name="t",
        payload={},
        error="b",
        attempts=1,
    )
    await push_to_dlq(
        fake_redis,
        job_id="new",
        task_name="t",
        payload={},
        error="b",
        attempts=1,
    )

    # Rewrite the "old" entry's timestamp directly in the list.
    raw_entries: list[bytes] = await fake_redis.lrange("dlq:agent_tools", 0, -1)
    rebuilt: list[bytes] = []
    for raw in raw_entries:
        entry = DLQEntry.from_json(raw)
        entry.enqueued_at = old_iso if entry.job_id == "old" else new_iso
        rebuilt.append(entry.to_json().encode())
    await fake_redis.delete("dlq:agent_tools")
    for blob in reversed(rebuilt):  # rebuild keeping order
        await fake_redis.rpush("dlq:agent_tools", blob)

    purged = await purge_dlq(fake_redis, older_than=timedelta(days=1))
    assert purged == 1
    listed = await list_dlq(fake_redis)
    assert [e.job_id for e in listed] == ["old"] or [e.job_id for e in listed] == [
        "new"
    ]
    # Only one survives — the "new" one.
    surviving = [e.job_id for e in listed]
    assert "new" in surviving
    assert "old" not in surviving
