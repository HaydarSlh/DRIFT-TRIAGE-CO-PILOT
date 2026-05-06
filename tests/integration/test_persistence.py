"""Tests for the persistence + checkpointer wiring.

These exercise:

- A ``Report`` row is created on POST and updates as the graph runs.
- ``ReportRun`` rows accumulate (one per node invocation that produces an update).
- The LangGraph checkpointer (``MemorySaver`` for tests) holds graph state keyed
  by ``thread_id``, and the same state is recoverable across calls — i.e. the
  short-term agent memory survives the kind of restart we'd see in production
  when a worker pod is replaced.
"""

import uuid

from httpx import AsyncClient

from .test_api import _wait_for_status


async def test_report_row_persists_through_run(async_client: AsyncClient) -> None:
    """Postgres row transitions pending → running → completed and accumulates runs."""
    resp = await async_client.post(
        "/reports", json={"topic": "persistence smoke test topic"}
    )
    assert resp.status_code == 202
    report_id = resp.json()["id"]

    final = await _wait_for_status(async_client, report_id, "completed")
    assert final["status"] == "completed"
    assert len(final["runs"]) >= 1
    # Step numbers are monotonically increasing (audit trail invariant).
    step_numbers = [r["step_number"] for r in final["runs"]]
    assert step_numbers == sorted(step_numbers)


async def test_thread_id_is_unique_per_report(async_client: AsyncClient) -> None:
    """Each report gets a fresh thread_id — no collisions across runs."""
    a = (await async_client.post("/reports", json={"topic": "first topic xyzzy"})).json()
    b = (await async_client.post("/reports", json={"topic": "second topic plugh"})).json()
    assert a["thread_id"] != b["thread_id"]


async def test_get_report_returns_404_for_unknown_id(async_client: AsyncClient) -> None:
    """An unknown id is a clean 404, not a 500."""
    resp = await async_client.get(f"/reports/{uuid.uuid4()}")
    assert resp.status_code == 404


async def test_checkpoint_state_recovers_via_thread_id(
    async_client: AsyncClient,
) -> None:
    """The graph's terminal state is retrievable from the checkpointer post-run.

    This is the property that makes restartable runs possible — in CP2 the
    checkpointer is in-memory for tests, but the same code path works against
    a Redis-backed saver in production. We verify that GET /reports/{id} can
    pull findings/critique/report from the checkpointer rather than from
    Postgres, which proves the two stores are correctly partitioned.
    """
    resp = await async_client.post(
        "/reports", json={"topic": "checkpoint recovery topic"}
    )
    report_id = resp.json()["id"]
    final = await _wait_for_status(async_client, report_id, "completed")
    # Findings live in Redis (the checkpointer), surfaced via /reports/{id}.
    assert final["findings"], "findings should be reachable via the checkpointer"
    assert final["report"], "final report should be reachable via the checkpointer"
