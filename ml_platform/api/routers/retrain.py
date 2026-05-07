import logging
import json
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient

from ml_platform.config import (
    TRAIN_PATH, TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS,
    PIPELINE_PATH, THRESHOLD_PATH, REFERENCE_STATS_PATH,
    MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, RANDOM_STATE,
    ARTIFACT_DIR
)
from ml_platform.models.train import build_pipeline  # reuse your pipeline builder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_recall_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Response model ---
class RetrainResponse(BaseModel):
    status: str
    model_version: int
    mlflow_run_id: str

# --- Drifted data generation (reuse from drifted_data.py) ---
def generate_drifted_train():
    """Creates a drifted version of the original train split."""
    train = pd.read_parquet(TRAIN_PATH)
    df = train.copy()

    # Shift euribor3m by +2 std
    std_euribor = df['euribor3m'].std()
    df['euribor3m'] = df['euribor3m'] + 2 * std_euribor

    # Shift 'job' to 'housemaid' for half the rows
    rng = np.random.default_rng(42)
    mask = rng.choice(df.index, size=len(df)//2, replace=False)
    df.loc[mask, 'job'] = 'housemaid'

    drifted_path = TRAIN_PATH.parent / "drifted_train.parquet"
    df.to_parquet(drifted_path, index=False)
    return drifted_path

# --- Training + registration helper ---
def train_and_register(drifted_data_path: Path):
    """Trains on the given dataset, registers the model, and returns version + run_id."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Load drifted training data and validation set (original val split is fine)
    train = pd.read_parquet(drifted_data_path)
    val = pd.read_parquet(TRAIN_PATH.parent / "val.parquet")   # keep val unchanged

    X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
    X_val, y_val = val.drop(columns=[TARGET_COL]), val[TARGET_COL]

    # Build pipeline and hyperparameter search (same as your train.py)
    pipeline = build_pipeline()
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,               # fewer iterations for faster demo
        scoring='roc_auc',
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    # Threshold tuning
    y_prob_val = best_pipeline.predict_proba(X_val)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_val, y_prob_val)
    meets_recall = rec[:-1] >= 0.75
    if meets_recall.any():
        idx = meets_recall.nonzero()[0][-1]
        op_threshold = float(thresh[idx])
    else:
        op_threshold = 0.5

    val_preds = (y_prob_val >= op_threshold).astype(int)
    val_auc = roc_auc_score(y_val, y_prob_val)
    val_recall = recall_score(y_val, val_preds)

    # Save artifacts
    joblib.dump(best_pipeline, PIPELINE_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": op_threshold, "val_recall": float(val_recall)}, f)

    # Log to MLflow and register a new version
    with mlflow.start_run(run_name="retrain_on_drifted") as run:
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_param("operating_threshold", op_threshold)

        run_id = run.info.run_id

    # Get the latest version created
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest_version = max(int(mv.version) for mv in versions)

    return latest_version, run_id

# --- Endpoint ---
@router.post("/retrain", response_model=RetrainResponse)
async def retrain():
    """
    Triggers a retrain using drifted data (simulated).
    This is called by the queue worker when the agent decides to retrain.
    """
    try:
        # Generate drifted data (or use a pre-existing file)
        drifted_data = generate_drifted_train()
        logger.info(f"Drifted training data saved to {drifted_data}")

        # Train and register
        new_version, run_id = train_and_register(drifted_data)
        logger.info(f"Retrain complete: new version {new_version}, run {run_id}")

        return RetrainResponse(
            status="success",
            model_version=new_version,
            mlflow_run_id=run_id
        )
    except Exception as e:
        logger.exception("Retrain failed")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")