"""``retrain`` — kick off a shadow retraining run via the platform.

Calls the platform's ``POST /models/train`` endpoint. Returns the new
candidate version. Permanent failures (4xx, schema mismatch) are not retried.
Transient failures (5xx, 429, connection errors) are retried.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.errors import PermanentToolError, TransientToolError
from app.core.logging import get_logger
from app.taskqueue import task

log = get_logger(__name__)


class RetrainInput(BaseModel):
    model_id: str = Field(min_length=1)
    dataset_version: str = Field(default="latest")


class RetrainOutput(BaseModel):
    model_id: str
    candidate_version: str
    training_run_id: str


@task("retrain_shadow", retry_on=(TransientToolError,), no_retry_on=(PermanentToolError,))
async def retrain_shadow(payload: dict[str, Any]) -> dict[str, Any]:
    """Kick off a shadow retraining run.

    Args:
        payload: ``{"model_id": "...", "dataset_version": "latest"}``.

    Returns:
        ``{"model_id", "candidate_version", "training_run_id"}``.
    """
    try:
        parsed = RetrainInput.model_validate(payload)
    except Exception as exc:
        raise PermanentToolError(f"invalid retrain payload: {exc}") from exc

    import httpx
    from app.config import get_settings

    settings = get_settings()
    platform_url = settings.platform_url

    log.info("retrain_started", model_id=parsed.model_id, dataset_version=parsed.dataset_version)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{platform_url}/models/train",
                json={"model_id": parsed.model_id, "dataset_version": parsed.dataset_version},
            )
    except httpx.ConnectError as exc:
        raise TransientToolError(f"retrain connect failed: {exc}") from exc
    except httpx.ReadTimeout as exc:
        raise TransientToolError(f"retrain timeout: {exc}") from exc
    except httpx.RequestError as exc:
        raise TransientToolError(f"retrain request error: {exc}") from exc

    if response.status_code >= 500 or response.status_code == 429:
        raise TransientToolError(f"retrain server returned {response.status_code}")
    if 400 <= response.status_code < 500:
        raise PermanentToolError(f"retrain client error {response.status_code}: {response.text}")

    data = response.json()
    return RetrainOutput(
        model_id=parsed.model_id,
        candidate_version=data.get("candidate_version", "unknown"),
        training_run_id=data.get("training_run_id", ""),
    ).model_dump()