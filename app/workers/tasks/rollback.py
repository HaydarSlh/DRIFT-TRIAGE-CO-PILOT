"""``rollback`` — revert Production to a previous registered model version.

Calls the platform's ``POST /registry/promote`` endpoint with ``target_version``.

Critical: the idempotency key must encode the *target version*, not just the
model_id, so two retries against different target versions don't collide.
The caller (comms_node via call_tool) passes an idempotency key that includes
the thread_id and action_type, which is sufficient for uniqueness.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.errors import PermanentToolError, TransientToolError
from app.core.logging import get_logger
from app.taskqueue import task

log = get_logger(__name__)


class RollbackInput(BaseModel):
    model_id: str = Field(min_length=1)
    target_version: str = Field(min_length=1, description="Version to roll back to.")


class RollbackOutput(BaseModel):
    model_id: str
    previous_version: str
    new_production_version: str


@task("rollback", retry_on=(TransientToolError,), no_retry_on=(PermanentToolError,))
async def rollback(payload: dict[str, Any]) -> dict[str, Any]:
    """Roll back Production to the previous registered version.

    Args:
        payload: ``{"model_id": "...", "target_version": "2"}``.

    Returns:
        ``{"model_id", "previous_version", "new_production_version"}``.
    """
    try:
        parsed = RollbackInput.model_validate(payload)
    except Exception as exc:
        raise PermanentToolError(f"invalid rollback payload: {exc}") from exc

    import httpx
    from app.config import get_settings

    settings = get_settings()
    platform_url = settings.platform_url

    log.info("rollback_started", model_id=parsed.model_id, target_version=parsed.target_version)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{platform_url}/rollback",
                json={
                    "action": "rollback",
                    "model_version": parsed.target_version,
                    "model_id": parsed.model_id,
                    "reason": f"Drift rollback to version {parsed.target_version}",
                },
            )
    except httpx.ConnectError as exc:
        raise TransientToolError(f"rollback connect failed: {exc}") from exc
    except httpx.ReadTimeout as exc:
        raise TransientToolError(f"rollback timeout: {exc}") from exc
    except httpx.RequestError as exc:
        raise TransientToolError(f"rollback request error: {exc}") from exc

    if response.status_code >= 500 or response.status_code == 429:
        raise TransientToolError(f"rollback server returned {response.status_code}")
    if 400 <= response.status_code < 500:
        raise PermanentToolError(f"rollback client error {response.status_code}: {response.text}")

    data = response.json()
    return RollbackOutput(
        model_id=parsed.model_id,
        previous_version=data.get("previous_version", ""),
        new_production_version=data.get("new_production_version", parsed.target_version),
    ).model_dump()