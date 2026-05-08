from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import logging
import mlflow
from mlflow.tracking import MlflowClient
from ml_platform.config import (
    TEST_PATH, TARGET_COL, MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME, THRESHOLD_PATH
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

router = APIRouter()
logger = logging.getLogger(__name__)

# Tolerance below baseline AUC at which the model is still considered passing.
AUC_TOLERANCE = 0.02


class ReplayRequest(BaseModel):
    model_id: str


class ReplayResponse(BaseModel):
    model_id: str
    auc: float
    baseline_auc: float
    passed: bool
    test_f1: float
    test_recall: float
    test_accuracy: float
    num_test_samples: int


@router.post("/models/replay", response_model=ReplayResponse)
async def replay_test(request: ReplayRequest):
    """Replay the held-out test set on the current Production model.

    Returns the test AUC plus the baseline AUC (logged at training time) and a
    pass/fail decision: ``passed = test_auc >= baseline_auc - AUC_TOLERANCE``.
    The agent's replay_test worker uses this to decide whether the production
    model has degraded against its own held-out test set.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        prod_versions = [
            v for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            if v.current_stage == "Production"
        ]
        if not prod_versions:
            raise HTTPException(status_code=503, detail="No Production model version registered")
        prod = sorted(prod_versions, key=lambda v: int(v.version))[-1]

        run = client.get_run(prod.run_id)
        baseline_auc = run.data.metrics.get("val_auc")
        if baseline_auc is None:
            raise HTTPException(
                status_code=409,
                detail=f"Production run {prod.run_id} is missing val_auc metric",
            )

        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{prod.version}")
        with open(THRESHOLD_PATH) as f:
            threshold = json.load(f)["threshold"]

        test = pd.read_parquet(TEST_PATH)
        X_test = test.drop(columns=[TARGET_COL])
        y_test = test[TARGET_COL]

        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= threshold).astype(int)

        test_auc = float(roc_auc_score(y_test, y_probs))
        passed = test_auc >= baseline_auc - AUC_TOLERANCE

        logger.info(
            f"Replay v{prod.version}: AUC={test_auc:.3f} baseline={baseline_auc:.3f} "
            f"passed={passed}"
        )

        return ReplayResponse(
            model_id=request.model_id,
            auc=test_auc,
            baseline_auc=float(baseline_auc),
            passed=passed,
            test_f1=float(f1_score(y_test, y_preds)),
            test_recall=float(recall_score(y_test, y_preds)),
            test_accuracy=float(accuracy_score(y_test, y_preds)),
            num_test_samples=len(y_test),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Replay test failed")
        raise HTTPException(status_code=500, detail=f"Replay failed: {str(e)}")