"""``replay_test`` — re-score the held-out test set against current Production.

Calls the platform's ``POST /predict/replay`` endpoint. Deterministic for tests
(the platform stub returns canned results). Raises TransientToolError on
5xx/429 and PermanentToolError on 4xx so the retry policy and DLQ behave
correctly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.errors import PermanentToolError, TransientToolError
from app.core.logging import get_logger
from app.taskqueue import task

log = get_logger(__name__)


class ReplayTestInput(BaseModel):
    model_id: str = Field(min_length=1)
    test_set: str = Field(default="heldout", description="Which test set to replay against.")


class ReplayTestOutput(BaseModel):
    model_id: str
    replay_auc: float
    baseline_auc: float
    passed: bool


@task("replay_test", retry_on=(TransientToolError,), no_retry_on=(PermanentToolError,))
async def replay_test(payload: dict[str, Any]) -> dict[str, Any]:
    """Replay the test set against the current production model.

    Args:
        payload: ``{"model_id": "...", "test_set": "heldout"}``.

    Returns:
        ``{"model_id", "replay_auc", "baseline_auc", "passed"}``.
    """
    try:
        parsed = ReplayTestInput.model_validate(payload)
    except Exception as exc:
        raise PermanentToolError(f"invalid replay_test payload: {exc}") from exc

    import httpx
    from app.config import get_settings

    settings = get_settings()
    platform_url = settings.platform_url

    log.info("replay_test_started", model_id=parsed.model_id, test_set=parsed.test_set)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{platform_url}/predict/replay",
                json={"model_id": parsed.model_id, "test_set": parsed.test_set},
            )
    except httpx.ConnectError as exc:
        raise TransientToolError(f"replay_test connect failed: {exc}") from exc
    except httpx.ReadTimeout as exc:
        raise TransientToolError(f"replay_test timeout: {exc}") from exc
    except httpx.RequestError as exc:
        raise TransientToolError(f"replay_test request error: {exc}") from exc

    if response.status_code >= 500 or response.status_code == 429:
        raise TransientToolError(f"replay_test server returned {response.status_code}")
    if 400 <= response.status_code < 500:
        raise PermanentToolError(f"replay_test client error {response.status_code}: {response.text}")

    data = response.json()
    return ReplayTestOutput(
        model_id=parsed.model_id,
        replay_auc=data.get("auc", 0.0),
        baseline_auc=data.get("baseline_auc", 0.0),
        passed=data.get("passed", False),
    ).model_dump()