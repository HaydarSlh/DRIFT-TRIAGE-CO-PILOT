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

    target_version = body.target_version

    if not target_version:
        prod_versions = [
            v for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            if v.current_stage == "Production"
        ]
        if not prod_versions:
            raise HTTPException(
                status_code=409,
                detail="No Production version to roll back from.",
            )
        current_prod = sorted(prod_versions, key=lambda v: int(v.version))[-1]
        candidate = str(int(current_prod.version) - 1)
        try:
            client.get_model_version(name=REGISTERED_MODEL_NAME, version=candidate)
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"Previous version {candidate} not found in registry.",
            )
        target_version = candidate

    try:
        target = client.get_model_version(name=REGISTERED_MODEL_NAME, version=target_version)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Target version {target_version} not found in registry.",
        )

    prod_versions = [
        v for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if v.current_stage == "Production" and v.version != target_version
    ]
    previous_version = (
        sorted(prod_versions, key=lambda v: int(v.version))[-1].version
        if prod_versions else ""
    )

    if target.current_stage == "Production" and not prod_versions:
        logger.info(f"Rollback no-op: v{target_version} is already the sole Production version")
        return RollbackResponse(
            model_id=body.model_id,
            previous_version="",
            new_production_version=target_version,
            mlflow_run_id=target.run_id or "",
        )

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=int(target_version),
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(
        f"Rollback executed: v{previous_version or '(none)'} → v{target_version} "
        f"(reason: {body.reason!r})"
    )

    return RollbackResponse(
        model_id=body.model_id,
        previous_version=previous_version,
        new_production_version=target_version,
        mlflow_run_id=target.run_id or "",
    )
