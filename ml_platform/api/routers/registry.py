from fastapi import APIRouter, HTTPException, Request
from datetime import datetime, timezone, timedelta
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import logging

from ml_platform.schemas import PromotionRequest, PromotionResponse
from ml_platform.config import (
    REGISTERED_MODEL_NAME,
    THRESHOLD_PATH,
    REFERENCE_STATS_PATH
    ,MLFLOW_TRACKING_URI
)

logger = logging.getLogger(__name__)
router = APIRouter()

PROMOTION_CONTRACT_VERSION = "promotion-v1"

# ------------------------------------------------------------
# Day‑4 Promotion Checklist
# ------------------------------------------------------------
def run_promotion_checklist(model_version: str, action: str) -> tuple[bool, str]:
    """
    Returns (passed: bool, failure_reason: str).
    Only applies for 'promote' actions; rollbacks pass automatically.
    """
    if action != "promote":
        return True, ""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Model version must exist
    try:
        mv = client.get_model_version(name=REGISTERED_MODEL_NAME, version=model_version)
    except Exception:
        return False, f"Model version {model_version} not found in registry."

    # 2. Training run must be linked
    run_id = mv.run_id
    if not run_id:
        return False, "No MLflow run linked to this model version."
    try:
        run = client.get_run(run_id)
    except Exception:
        return False, f"MLflow run {run_id} not found."

    # 3. Operating threshold file must exist
    if not THRESHOLD_PATH.exists():
        return False, "Operating threshold file missing."

    # 4. Metrics from the training run
    val_auc = run.data.metrics.get("val_auc") if run.data.metrics else None
    val_recall = run.data.metrics.get("val_recall") if run.data.metrics else None

    if val_auc is None or val_auc < 0.75:
        return False, f"Validation AUC ({val_auc}) is below 0.75 or missing."
    if val_recall is None or val_recall < 0.75:
        return False, f"Validation recall ({val_recall}) is below 0.75 or missing."

    # 5. Model fidelity snapshot (proof that replay test passed)
    snapshot_path = Path("tests/test_probs_snapshot.npy")
    if not snapshot_path.exists():
        return False, "Model fidelity snapshot missing. Run the fidelity test first."

    # 6. Reference statistics must be frozen
    if not REFERENCE_STATS_PATH.exists():
        return False, "Reference statistics file missing."

    # All checks passed
    return True, ""

@router.post("/registry/promote", response_model=PromotionResponse)
async def promote(request: Request, body: PromotionRequest):
    # 1. Validate contract version header
    version_header = request.headers.get("X-Contract-Version")
    if version_header != PROMOTION_CONTRACT_VERSION:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported contract version. Expected {PROMOTION_CONTRACT_VERSION}"
        )

    # 2. Reject expired approvals (older than 1 hour)
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
    new_prod = body.model_version

    if body.action == "promote":
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=int(body.model_version),
            stage="Production",
            archive_existing_versions=True
        )
    else:  # rollback
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=int(body.model_version),
            stage="Production",
            archive_existing_versions=True
        )

    logger.info(f"Promotion executed: {body.action} v{body.model_version} by {body.approved_by}")

    return PromotionResponse(
        status="promoted" if body.action == "promote" else "rolled_back",
        new_production_version=new_prod,
        mlflow_run_id="N/A"   # could be retrieved from client
    )