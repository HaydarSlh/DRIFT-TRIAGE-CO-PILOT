"""Thin HTTP clients for the agent service, the ml_platform service, and MLflow.

Why a single module: every page imports from here, so retries, timeouts, and the
"unreachable-service" fallback live in one place. Streamlit's `st.cache_data`
wraps the read-side calls so we don't hammer the backends every rerun.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import streamlit as st

AGENT_URL = os.environ.get("AGENT_URL", "http://agent:8001")
PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://platform:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://platform:5000")
PROMOTION_CONTRACT = "promotion-v1"
DEFAULT_TIMEOUT = 8.0


class ApiError(Exception):
    """Raised when a backend returns a non-2xx response we want to surface."""


def _get(url: str, **kwargs: Any) -> Any:
    try:
        r = httpx.get(url, timeout=DEFAULT_TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise ApiError(f"{e.response.status_code} {e.response.text[:200]}") from e
    except httpx.HTTPError as e:
        raise ApiError(str(e)) from e


def _post(url: str, json_body: dict | None = None, timeout: float = DEFAULT_TIMEOUT, **kwargs: Any) -> Any:
    try:
        r = httpx.post(url, json=json_body, timeout=timeout, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise ApiError(f"{e.response.status_code} {e.response.text[:200]}") from e
    except httpx.HTTPError as e:
        raise ApiError(str(e)) from e


def _delete(url: str, **kwargs: Any) -> Any:
    try:
        r = httpx.delete(url, timeout=DEFAULT_TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise ApiError(f"{e.response.status_code} {e.response.text[:200]}") from e
    except httpx.HTTPError as e:
        raise ApiError(str(e)) from e


# ---------------------------------------------------------------------------
# Agent service
# ---------------------------------------------------------------------------


@st.cache_data(ttl=5, show_spinner=False)
def agent_health() -> dict[str, Any]:
    return _get(f"{AGENT_URL}/health/ready")


@st.cache_data(ttl=5, show_spinner=False)
def agent_metrics() -> dict[str, Any]:
    return _get(f"{AGENT_URL}/health/metrics")


@st.cache_data(ttl=5, show_spinner=False)
def list_investigations() -> list[dict[str, Any]]:
    return _get(f"{AGENT_URL}/investigations") or []


@st.cache_data(ttl=3, show_spinner=False)
def get_investigation(inv_id: str) -> dict[str, Any]:
    return _get(f"{AGENT_URL}/investigations/{inv_id}")


@st.cache_data(ttl=3, show_spinner=False)
def list_pending_approvals() -> list[dict[str, Any]]:
    return _get(f"{AGENT_URL}/approvals/pending") or []


def respond_to_approval(approval_id: str, approved: bool, feedback: str) -> dict[str, Any]:
    return _post(
        f"{AGENT_URL}/approvals/{approval_id}/respond",
        json_body={"approved": approved, "feedback": feedback},
        timeout=120.0,  # graph resume + LLM calls + tool dispatch can take ~60s
    )


@st.cache_data(ttl=3, show_spinner=False)
def get_dlq(limit: int = 100) -> dict[str, Any]:
    return _get(f"{AGENT_URL}/jobs/dlq/list", params={"limit": limit})


@st.cache_data(ttl=3, show_spinner=False)
def get_job(job_id: str) -> dict[str, Any]:
    return _get(f"{AGENT_URL}/jobs/{job_id}")


def requeue_dlq(job_id: str) -> dict[str, Any]:
    return _post(f"{AGENT_URL}/jobs/dlq/{job_id}/requeue")


def delete_dlq(job_id: str) -> dict[str, Any]:
    return _delete(f"{AGENT_URL}/jobs/dlq/{job_id}")


def purge_dlq(older_than_seconds: int | None = None) -> dict[str, Any]:
    params = {}
    if older_than_seconds is not None:
        params["older_than_seconds"] = older_than_seconds
    return _post(f"{AGENT_URL}/jobs/dlq/purge", json_body=None, params=params)


# ---------------------------------------------------------------------------
# Platform service (ml_platform)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def platform_health() -> dict[str, Any]:
    try:
        return _get(f"{PLATFORM_URL}/health")
    except ApiError:
        return _get(f"{PLATFORM_URL}/")


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    return _post(f"{PLATFORM_URL}/predict", json_body=payload)


def trigger_retrain() -> dict[str, Any]:
    return _post(f"{PLATFORM_URL}/retrain")


def promote_version(model_version: str, action: str, approved_by: str, approval_ts: str) -> dict[str, Any]:
    headers = {"X-Contract-Version": PROMOTION_CONTRACT}
    return _post(
        f"{PLATFORM_URL}/registry/promote",
        json_body={
            "model_version": model_version,
            "action": action,
            "approved_by": approved_by,
            "approval_timestamp": approval_ts,
            "investigation_id": str(uuid.uuid4()),
            "reason": f"Manual {action} from dashboard by {approved_by}",
        },
        headers=headers,
    )


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def list_registered_models() -> list[dict[str, Any]]:
    """Direct MLflow REST: /api/2.0/mlflow/registered-models/search (MLflow 2.x)."""
    try:
        data = _get(f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search")
        return data.get("registered_models", []) or []
    except ApiError:
        return []


@st.cache_data(ttl=10, show_spinner=False)
def list_model_versions(name: str) -> list[dict[str, Any]]:
    try:
        data = _get(
            f"{MLFLOW_URL}/api/2.0/mlflow/model-versions/search",
            params={"filter": f"name='{name}'"},
        )
        return data.get("model_versions", []) or []
    except ApiError:
        return []


@st.cache_data(ttl=10, show_spinner=False)
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return _get(f"{MLFLOW_URL}/api/2.0/mlflow/runs/get", params={"run_id": run_id})
    except ApiError:
        return {}
