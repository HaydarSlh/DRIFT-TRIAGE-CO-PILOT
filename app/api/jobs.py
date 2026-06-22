"""Job + DLQ admin endpoints.

Public endpoints:
    - GET    /jobs/{job_id}           — current status + result
Admin endpoints:
    - GET    /jobs/dlq                — newest 100 DLQ entries
    - POST   /jobs/dlq/{job_id}/requeue
    - DELETE /jobs/dlq/{job_id}       — purge a single entry
    - POST   /jobs/dlq/purge          — purge older-than (admin sweep)

These are deliberately small. The trace-replay-fix loop students learn in
CP3 still applies: a DLQ item is not a problem to be fire-and-forget
requeued — it is a question. "Why did this fail?" comes before "let me try
again." We keep requeue cheap so the workflow is fast, but the workflow
itself is "investigate, then act."
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.errors import ToolError
from app.core.logging import get_logger
from app.deps import build_queue_client, get_queue_client_dep
from app.taskqueue.client import QueueClient
from app.taskqueue.dlq import (
    get_dlq_size,
    list_dlq,
    purge_dlq,
    requeue_from_dlq,
)
from app.taskqueue.results import JobStatus, load_result

router = APIRouter(prefix="/jobs", tags=["jobs"])
log = get_logger(__name__)

QueueDep = Annotated[QueueClient | None, Depends(get_queue_client_dep)]


async def _require_queue(queue: QueueClient | None) -> QueueClient:
    """Construct a queue client on demand if app.state didn't get one."""
    if queue is not None:
        return queue
    try:
        return await build_queue_client()
    except Exception as exc:
        raise HTTPException(503, f"queue unavailable: {exc}") from exc


@router.get("/{job_id}")
async def get_job(job_id: str, queue: QueueDep) -> dict[str, Any]:
    """Return the current status (and result, if any) for ``job_id``."""
    qc = await _require_queue(queue)
    status = await qc.get_status(job_id)
    result = await load_result(qc._redis, job_id)
    payload: dict[str, Any] = {"job_id": job_id, "status": status.value}
    if result is not None:
        payload["attempts"] = result.attempts
        payload["completed_at"] = result.completed_at
        if status == JobStatus.COMPLETED:
            payload["result_data"] = result.result_data
        elif result.error:
            payload["error"] = result.error
    return payload


@router.get("/dlq/list")
async def get_dlq(queue: QueueDep, limit: int = 100) -> dict[str, Any]:
    """List newest DLQ entries."""
    qc = await _require_queue(queue)
    entries = await list_dlq(qc._redis, limit=limit)
    size = await get_dlq_size(qc._redis)
    return {
        "size": size,
        "entries": [
            {
                "job_id": e.job_id,
                "task_name": e.task_name,
                "payload": e.payload,
                "error": e.error,
                "attempts": e.attempts,
                "enqueued_at": e.enqueued_at,
            }
            for e in entries
        ],
    }


@router.post("/dlq/{job_id}/requeue")
async def requeue_job(job_id: str, queue: QueueDep) -> dict[str, Any]:
    """Pop the matching DLQ entry and re-enqueue under the same idempotency key."""
    qc = await _require_queue(queue)
    new_id = await requeue_from_dlq(qc._redis, job_id=job_id, queue_client=qc)
    if new_id is None:
        raise HTTPException(404, f"no DLQ entry for job_id={job_id}")
    return {"old_job_id": job_id, "new_job_id": new_id}


@router.delete("/dlq/{job_id}")
async def delete_dlq_entry(job_id: str, queue: QueueDep) -> dict[str, Any]:
    """Remove a single DLQ entry without re-enqueuing.

    The DLQ list is small, so a linear scan + LREM is fine.
    """
    qc = await _require_queue(queue)
    entries = await list_dlq(qc._redis, limit=10_000)
    for raw_idx, entry in enumerate(entries):
        if entry.job_id == job_id:
            # Reconstruct the same JSON the comms agent used (dataclasses round-trip
            # fine because field order is fixed) and LREM it.
            await qc._redis.lrem("dlq:agent_tools", 1, entry.to_json())
            log.info("dlq_entry_deleted", job_id=job_id, position=raw_idx)
            return {"job_id": job_id, "deleted": True}
    raise HTTPException(404, f"no DLQ entry for job_id={job_id}")


@router.post("/dlq/purge")
async def purge_dlq_endpoint(
    queue: QueueDep,
    older_than_seconds: int | None = None,
) -> dict[str, Any]:
    """Purge old DLQ entries. ``older_than_seconds=None`` purges everything."""
    qc = await _require_queue(queue)
    cutoff: timedelta | None = (
        None if older_than_seconds is None else timedelta(seconds=older_than_seconds)
    )
    purged = await purge_dlq(qc._redis, older_than=cutoff)
    return {"purged": purged}


# Re-export ToolError so the router module is the API-error boundary.
__all__ = ["ToolError", "router"]
