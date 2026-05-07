"""Drift webhook endpoint — receives alerts from the platform.

The platform POSTs drift events here. The agent creates an investigation
(in the PENDING state), hands the graph run to a background task, and returns 202.

Contract versioning: the platform must send ``X-Contract-Version: drift-alert-v1``.
Unrecognized or missing version → 400.
"""

import hashlib
import hmac
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Header, Request

from app.core.logging import get_logger
from app.deps import get_investigation_service
from app.schemas.webhooks import DriftAlertAccepted, DriftAlertEvent
from app.services.investigation_service import InvestigationService

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
log = get_logger(__name__)

InvestigationServiceDep = Annotated[InvestigationService, Depends(get_investigation_service)]


def _verify_hmac(request_body: bytes, signature: str | None, secret: str) -> bool:
    """Verify HMAC-SHA256 signature of the request body."""
    if not signature or not secret:
        return False
    expected = hmac.new(secret.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


@router.post("/drift", response_model=DriftAlertAccepted, status_code=202)
async def on_drift_event(
    event: DriftAlertEvent,
    background_tasks: BackgroundTasks,
    service: InvestigationServiceDep,
    request: Request,
    x_contract_version: Annotated[str | None, Header()] = None,
) -> DriftAlertAccepted:
    """Receive a drift alert from the platform and start an investigation.

    Idempotent: if an investigation with the same ``event_id`` already exists,
    return 200 with ``duplicate=True`` instead of creating a new one.
    """
    if x_contract_version != "drift-alert-v1":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported contract version: {x_contract_version}. Expected: drift-alert-v1",
        )

    settings = request.app.state.settings if hasattr(request.app.state, "settings") else None
    if settings and hasattr(settings, "platform_webhook_secret") and settings.platform_webhook_secret:
        body = await request.body()
        sig = request.headers.get("X-HMAC-Signature")
        raw = settings.platform_webhook_secret
        if raw is None:
            secret = None
        elif hasattr(raw, "get_secret_value"):
            secret = raw.get_secret_value()
        else:
            secret = str(raw)
        if not _verify_hmac(body, sig, secret):
            raise HTTPException(status_code=401, detail="Invalid HMAC signature")

    investigation, is_duplicate = await service.start_from_drift_event(event)

    if is_duplicate:
        return DriftAlertAccepted(investigation_id=investigation.id, duplicate=True)

    background_tasks.add_task(
        service.run_in_background,
        investigation.id,
        event,
        requires_human_review=None,
    )
    return DriftAlertAccepted(investigation_id=investigation.id, duplicate=False)