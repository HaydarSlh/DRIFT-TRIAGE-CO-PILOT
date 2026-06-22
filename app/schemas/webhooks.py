"""Wire-level Pydantic schemas for the drift webhook endpoint.

This is the shared contract with the platform service. Schema changes are
**breaking** — the platform must send ``X-Contract-Version: drift-alert-v1``
and the agent must reject unrecognized versions with 400.
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DriftAlertEvent(BaseModel):
    """Incoming drift alert webhook payload from the platform.

    ``event_id`` is the idempotency key — duplicate redeliveries with the same
    ``event_id`` must be no-ops at every layer.
    """

    schema_version: Literal["1.0"] = "1.0"
    timestamp: datetime
    event_id: str = Field(description="Unique idempotency key for the webhook event.")
    model_id: str
    model_version: str = ""
    drift_report_id: str = ""
    severity: str = Field(description="Platform-assessed severity: green, yellow, or red.")
    psi: dict[str, float] = Field(default_factory=dict)
    chi2: dict[str, float] = Field(default_factory=dict)
    output_drift: float = 0.0


class DriftAlertAccepted(BaseModel):
    """Response for a successfully accepted drift alert webhook."""

    investigation_id: uuid.UUID
    duplicate: bool = False