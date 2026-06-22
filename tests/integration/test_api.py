"""End-to-end tests for the HTTP layer.

CP2's contract is async: POST /reports returns 202 with an id, the actual run
happens in a FastAPI BackgroundTask, and clients poll GET /reports/{id} for
status. Each test below polls with a short backoff.
"""

import asyncio

from httpx import AsyncClient


async def _wait_for_status(
    client: AsyncClient,
    report_id: str,
    target: str,
    *,
    timeout_s: float = 5.0,
) -> dict:
    """Poll GET /reports/{id} until ``status == target`` or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    last: dict = {}
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/reports/{report_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] == target:
            return last
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"report {report_id} stuck at status={last.get('status')}, expected {target}"
    )


async def test_health_ok(async_client: AsyncClient) -> None:
    """Liveness endpoint always returns ok without touching the LLM."""
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_llm_with_fake(async_client: AsyncClient) -> None:
    """Readiness probe goes through the fake LLM and reports latency."""
    resp = await async_client.get("/health/llm")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["provider"] == "anthropic"
    assert "latency_ms" in body


async def test_create_report_returns_202(async_client: AsyncClient) -> None:
    """POST returns 202 with id, status=pending, and a thread_id."""
    resp = await async_client.post(
        "/reports", json={"topic": "the economic impact of remote work"}
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "id" in body
    assert body["status"] == "pending"
    assert body["thread_id"]


async def test_create_report_rejects_short_topic(async_client: AsyncClient) -> None:
    """Pydantic validation rejects topics shorter than 3 chars before any work runs."""
    resp = await async_client.post("/reports", json={"topic": "x"})
    assert resp.status_code == 422


async def test_create_report_then_complete(async_client: AsyncClient) -> None:
    """Full async path: 202 → background graph run → completed with artifacts."""
    resp = await async_client.post("/reports", json={"topic": "test topic for cp2 e2e"})
    assert resp.status_code == 202
    report_id = resp.json()["id"]

    final = await _wait_for_status(async_client, report_id, "completed")
    assert final["topic"] == "test topic for cp2 e2e"
    assert final["report"], "completed report must have a report body"
    assert len(final["findings"]) >= 1
    # At least one ReportRun row was persisted per agent (4 nodes, ~4-7 steps).
    assert len(final["runs"]) >= 1


async def test_list_reports(async_client: AsyncClient) -> None:
    """GET /reports returns the rows we've created so far."""
    create = await async_client.post("/reports", json={"topic": "list test topic"})
    assert create.status_code == 202
    listing = await async_client.get("/reports")
    assert listing.status_code == 200
    rows = listing.json()
    assert any(r["id"] == create.json()["id"] for r in rows)
