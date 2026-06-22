"""Queue-test helpers.

The ``fake_redis`` fixture is defined at the top level (``tests/conftest.py``)
so it's available to every test file. This conftest exists to host
``StubArqPool``, the in-process arq replacement used by ``QueueClient``
unit tests.
"""

from __future__ import annotations

from typing import Any


class StubArqPool:
    """Just-enough arq replacement for QueueClient unit tests.

    Records every ``enqueue_job`` call and (optionally) refuses duplicates
    so we can drill the producer-side idempotency check.
    """

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.dedup: set[str] = set()

    async def enqueue_job(
        self, task_name: str, payload: dict[str, Any], **kwargs: Any
    ) -> Any:
        job_id = kwargs.get("_job_id")
        if job_id and job_id in self.dedup:
            return None  # arq returns None for "already queued"
        if job_id:
            self.dedup.add(job_id)
        self.enqueued.append((task_name, payload, kwargs))

        class _Job:
            def __init__(self, jid: str | None) -> None:
                self.job_id = jid

        return _Job(job_id) if job_id else _Job("auto-id-1")

    async def abort_job(self, _job_id: str) -> bool:
        return True

    async def exists(self, _key: str) -> bool:  # pragma: no cover
        return False
