"""Training pipeline — trains model, logs threshold, registers with MLflow."""
from __future__ import annotations

import mlflow


def train_pipeline(X_train, y_train, params: dict) -> str:
    """Train a model and return the MLflow run ID."""
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        # TODO: fit model, log metrics, log artifact
        return run.info.run_id


def compute_threshold(y_true, y_prob, metric: str = "f1") -> float:
    """Find the optimal decision threshold."""
    raise NotImplementedError
