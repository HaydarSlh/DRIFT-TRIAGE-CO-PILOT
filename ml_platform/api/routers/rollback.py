"""Rollback router — demote current Production and promote a target version.

Distinct from /registry/promote which carries the full approval flow
(investigation_id, approved_by, approval_timestamp, day-4 checklist).
Rollback is the emergency lever: agent identifies a regression, picks a
known-good prior version, calls this endpoint, done.
"""
import logging

import mlflow
from fastapi import APIRouter, HTTPException, Request
from mlflow.tracking import MlflowClient

from ml_platform.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME
from ml_platform.schemas import RollbackRequest, RollbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()

ROLLBACK_CONTRACT_VERSION = "rollback-v1"


@router.post("/registry/rollback", response_model=RollbackResponse)
async def rollback(request: Request, body: RollbackRequest):
    version_header = request.headers.get("X-Contract-Version")
    if version_header != ROLLBACK_CONTRACT_VERSION:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported contract version. Expected {ROLLBACK_CONTRACT_VERSION}",
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        target = client.get_model_version(name=REGISTERED_MODEL_NAME, version=body.target_version)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Target version {body.target_version} not found in registry.",
        )

    prod_versions = [
        v for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if v.current_stage == "Production" and v.version != body.target_version
    ]
    previous_version = (
        sorted(prod_versions, key=lambda v: int(v.version))[-1].version
        if prod_versions else ""
    )

    if target.current_stage == "Production" and not prod_versions:
        # Already the only production version — nothing to do.
        logger.info(f"Rollback no-op: v{body.target_version} is already the sole Production version")
        return RollbackResponse(
            model_id=body.model_id,
            previous_version="",
            new_production_version=body.target_version,
            mlflow_run_id=target.run_id or "",
        )

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=int(body.target_version),
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(
        f"Rollback executed: v{previous_version or '(none)'} → v{body.target_version} "
        f"(reason: {body.reason!r})"
    )

    return RollbackResponse(
        model_id=body.model_id,
        previous_version=previous_version,
        new_production_version=body.target_version,
        mlflow_run_id=target.run_id or "",
    )
