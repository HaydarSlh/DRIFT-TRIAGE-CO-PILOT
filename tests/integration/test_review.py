"""Tests for human-in-the-loop review + resume.

Flow under test:

1. POST /reports with ``requires_human_review=true``.
2. The graph runs through researcher and critic, then pauses at writer's
   dynamic interrupt. The Report row transitions to ``awaiting_review`` and a
   ReviewRequest row appears in ``pending``.
3. GET /reviews/pending returns that row.
4. POST /reviews/{id}/respond with approval resumes the graph.
5. The graph completes and the Report row transitions to ``completed`` with
   a populated report body.

Rejection follows the same path but ends with a stub report and the writer's
"rejected" branch.
"""

from httpx import AsyncClient

from .test_api import _wait_for_status


async def test_approval_resumes_and_completes(async_client: AsyncClient) -> None:
    """Approving a pending review drives the run to completion."""
    create = await async_client.post(
        "/reports",
        json={"topic": "human review approval flow", "requires_human_review": True},
    )
    assert create.status_code == 202
    report_id = create.json()["id"]

    await _wait_for_status(async_client, report_id, "awaiting_review")

    pending = await async_client.get("/reviews/pending")
    assert pending.status_code == 200
    rows = pending.json()
    assert any(r["report_id"] == report_id for r in rows), "review should appear in pending list"
    review_id = next(r["id"] for r in rows if r["report_id"] == report_id)

    respond = await async_client.post(
        f"/reviews/{review_id}/respond",
        json={"approved": True, "feedback": "looks good"},
    )
    assert respond.status_code == 200
    body = respond.json()
    assert body["status"] == "approved"

    final = await _wait_for_status(async_client, report_id, "completed")
    assert final["report"], "approved run must produce a report body"
    assert "rejected" not in (final["report"] or "").lower()


async def test_rejection_records_stub_report(async_client: AsyncClient) -> None:
    """A rejected review still completes the run — with a stub rejection report."""
    create = await async_client.post(
        "/reports",
        json={"topic": "human review rejection flow", "requires_human_review": True},
    )
    report_id = create.json()["id"]
    await _wait_for_status(async_client, report_id, "awaiting_review")

    pending = await async_client.get("/reviews/pending")
    review_id = next(r["id"] for r in pending.json() if r["report_id"] == report_id)

    respond = await async_client.post(
        f"/reviews/{review_id}/respond",
        json={"approved": False, "feedback": "findings are too thin"},
    )
    assert respond.status_code == 200
    assert respond.json()["status"] == "rejected"

    final = await _wait_for_status(async_client, report_id, "completed")
    # The stub report is short and contains the reviewer's feedback verbatim.
    assert "rejected" in (final["report"] or "").lower()
    assert "findings are too thin" in (final["report"] or "")


async def test_resolved_review_cannot_be_resolved_twice(
    async_client: AsyncClient,
) -> None:
    """Posting to /respond on a non-pending review is a 400, not a silent retry."""
    create = await async_client.post(
        "/reports",
        json={"topic": "double resolve test topic", "requires_human_review": True},
    )
    report_id = create.json()["id"]
    await _wait_for_status(async_client, report_id, "awaiting_review")

    pending = await async_client.get("/reviews/pending")
    review_id = next(r["id"] for r in pending.json() if r["report_id"] == report_id)

    first = await async_client.post(
        f"/reviews/{review_id}/respond",
        json={"approved": True, "feedback": "first response"},
    )
    assert first.status_code == 200

    second = await async_client.post(
        f"/reviews/{review_id}/respond",
        json={"approved": False, "feedback": "second response"},
    )
    assert second.status_code == 400
