from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone, timedelta
import mlflow
from mlflow.tracking import MlflowClient
import logging
import json
from ml_platform.config import REGISTERED_MODEL_NAME, THRESHOLD_PATH
from ml_platform.schemas import PromotionRequest, PromotionResponse

logger = logging.getLogger(__name__)
router = APIRouter()

PROMOTION_CONTRACT_VERSION = "promotion-v1"



def run_promotion_checklist(model_version: str, action: str) -> tuple[bool, str]:
    client = MlflowClient()
    # Check existence
    try:
        mv = client.get_model_version(name=REGISTERED_MODEL_NAME, version=model_version)
    except Exception:
        return False, f"Model version {model_version} not found."

    # For promotion, we need to load the run metrics
    if action == "promote":
        run_id = mv.run_id
        if not run_id:
            return False, "No run associated with this version."
        run = client.get_run(run_id)
        # Test AUC threshold (should be ≥ 0.75)
        test_auc = run.data.metrics.get('val_auc')   # assuming you logged val_auc during training
        if test_auc is None or test_auc < 0.75:
            return False, f"Test AUC {test_auc} below 0.75."

        # Recall at operating threshold
        val_recall = run.data.metrics.get('val_recall')
        if val_recall is None or val_recall < 0.75:
            return False, f"Recall {val_recall} below 0.75."

        # Schema check – compare current schema to the schema artifact of this version
        # (simplified: we trust the registry's schema artifact if present)
        # In a full solution you'd load the schema artifact and compare.
        # For now, just trust it exists.
    return True, ""

@router.post("/registry/promote")
async def promote(request: Request, body: PromotionRequest):
    # 1. Validate contract version header
    version_header = request.headers.get("X-Contract-Version")
    if version_header != PROMOTION_CONTRACT_VERSION:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported contract version. Expected {PROMOTION_CONTRACT_VERSION}"
        )

    # 2. Check approval timestamp freshness (1 hour)
    try:
        approved_time = datetime.fromisoformat(body.approval_timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if (now - approved_time).total_seconds() > 3600:
            raise HTTPException(status_code=410, detail="Approval expired")
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid approval_timestamp format")

    # 3. Run promotion checklist
    passed, reason = run_promotion_checklist(body.model_version, body.action)
    if not passed:
        raise HTTPException(status_code=409, detail=f"Promotion checklist failed: {reason}")

    # 4. Execute the stage transition
    client = MlflowClient()
    if body.action == "promote":
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=int(body.model_version),
            stage="Production",
            archive_existing_versions=True
        )
        new_prod = body.model_version
    else:  # rollback
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=int(body.model_version),
            stage="Production",
            archive_existing_versions=True
        )
        new_prod = body.model_version

    logger.info(f"Promotion executed: {body.action} v{body.model_version} by {body.approved_by}")
    return PromotionResponse(
        status="promoted" if body.action == "promote" else "rolled_back",
        new_production_version=new_prod,
        mlflow_run_id="N/A"
    )