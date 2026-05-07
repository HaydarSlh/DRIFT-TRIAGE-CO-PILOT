import mlflow
from mlflow.tracking import MlflowClient
from ml_platform.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

def log_missing_metrics():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Get the run that generated version 1
    mv = client.get_model_version(name=REGISTERED_MODEL_NAME, version="1")
    run_id = mv.run_id
    print(f"Found model version 1, run_id = {run_id}")

    # Log the metrics you already computed during training
    # (use the values printed by your train.py – e.g. Test AUC 0.780, Recall 0.767)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("val_auc", 0.780)      # ← replace with your actual Test AUC
        mlflow.log_metric("val_recall", 0.767)   # ← replace with your actual Test Recall

    # Verify they are now visible
    run = client.get_run(run_id)
    print("Metrics after update:", run.data.metrics)

if __name__ == "__main__":
    log_missing_metrics()