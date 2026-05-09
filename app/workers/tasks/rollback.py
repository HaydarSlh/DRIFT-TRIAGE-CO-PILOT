"""``rollback`` — revert Production to a previous registered model version.

Calls the platform's ``POST /registry/rollback`` endpoint (contract: rollback-v1).
The platform identifies the current Production version, demotes it, and
promotes ``target_version`` in its place.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.errors import PermanentToolError, TransientToolError
from app.core.logging import get_logger
from app.taskqueue import task

log = get_logger(__name__)

ROLLBACK_CONTRACT = "rollback-v1"


class RollbackInput(BaseModel):
    model_id: str = Field(min_length=1)
    target_version: str | None = Field(
        default=None,
        description="Version to roll back to. If omitted, rolls back 1 version behind current Production.",
    )


class RollbackOutput(BaseModel):
    model_id: str
    previous_version: str
    new_production_version: str


@task("rollback", retry_on=(TransientToolError,), no_retry_on=(PermanentToolError,))
async def rollback(payload: dict[str, Any]) -> dict[str, Any]:
    """Roll back Production to the previous registered version.

    Args:
        payload: ``{"model_id": "...", "target_version": "2"}``. ``target_version``
            is optional; when omitted, the platform rolls back 1 version behind
            the current Production version (404 if that version doesn't exist).

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

    target_for_request = parsed.target_version or ""
    reason = (
        f"Drift rollback to version {parsed.target_version}"
        if parsed.target_version
        else "Drift rollback to version 1 behind current Production"
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{platform_url}/registry/rollback",
                json={
                    "model_id": parsed.model_id,
                    "target_version": target_for_request,
                    "reason": reason,
                },
                headers={"X-Contract-Version": ROLLBACK_CONTRACT},
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
        previous_version=str(data.get("previous_version", "")),
        new_production_version=str(data["new_production_version"]),
    ).model_dump()