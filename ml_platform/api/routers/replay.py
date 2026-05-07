from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import logging
import mlflow
from ml_platform.config import (
    TEST_PATH, TARGET_COL, MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME, THRESHOLD_PATH
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

router = APIRouter()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------
class ReplayRequest(BaseModel):
    model_id: str

class ReplayResponse(BaseModel):
    model_id: str
    test_auc: float
    test_f1: float
    test_recall: float
    test_accuracy: float
    num_test_samples: int

# ------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------
@router.post("/models/replay", response_model=ReplayResponse)
async def replay_test(request: ReplayRequest):
    """
    Replay the holdout test set on the current Production model.
    Called by the agent's worker when the triage recommends 'replay_test'.
    Returns the performance metrics.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # 1. Load the current Production model
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")

        # 2. Load the operating threshold
        with open(THRESHOLD_PATH) as f:
            threshold = json.load(f)["threshold"]

        # 3. Load the frozen test set
        test = pd.read_parquet(TEST_PATH)
        X_test = test.drop(columns=[TARGET_COL])
        y_test = test[TARGET_COL]

        # 4. Predict and compute metrics
        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= threshold).astype(int)

        test_auc = float(roc_auc_score(y_test, y_probs))
        test_f1 = float(f1_score(y_test, y_preds))
        test_recall = float(recall_score(y_test, y_preds))
        test_acc = float(accuracy_score(y_test, y_preds))

        logger.info(
            f"Replay completed: AUC={test_auc:.3f}, F1={test_f1:.3f}, "
            f"Recall={test_recall:.3f}, Accuracy={test_acc:.3f}"
        )

        return ReplayResponse(
            model_id=request.model_id,
            test_auc=test_auc,
            test_f1=test_f1,
            test_recall=test_recall,
            test_accuracy=test_acc,
            num_test_samples=len(y_test)
        )
    except Exception as e:
        logger.exception("Replay test failed")
        raise HTTPException(status_code=500, detail=f"Replay failed: {str(e)}")