"""HTTP client for calling back into the platform service."""
import os

import httpx

_BASE = os.environ.get("PLATFORM_URL", "http://localhost:8000")


def notify(model_id: str, decision: str, action: str | None) -> None:
    httpx.post(
        f"{_BASE}/drift/notify",
        json={"model_id": model_id, "decision": decision, "action": action},
        timeout=10,
    ).raise_for_status()
