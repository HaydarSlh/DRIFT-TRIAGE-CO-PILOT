import hashlib
import json
import platform as py_platform

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score

# Hardcoded settings (the container sees these paths)
MLFLOW_TRACKING_URI = "sqlite:////app/mlruns/mlflow.db"
REGISTERED_MODEL_NAME = "BankMarketingClassifier"
PIPELINE_PATH = "/app/artifacts/pipeline.pkl"
THRESHOLD_PATH = "/app/artifacts/threshold.json"
VAL_PATH = "/app/data/val.parquet"
TARGET_COL = "y"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Load the already-trained pipeline and threshold
pipeline = joblib.load(PIPELINE_PATH)
with open(THRESHOLD_PATH) as f:
    threshold_data = json.load(f)

# Compute real val_auc against the held-out validation set
val = pd.read_parquet(VAL_PATH)
X_val = val.drop(columns=[TARGET_COL])
y_val = val[TARGET_COL]
y_probs = pipeline.predict_proba(X_val)[:, 1]
val_auc = float(roc_auc_score(y_val, y_probs))
val_recall = float(threshold_data["val_recall"])  # saved by train.py at threshold-tuning time

# Build schema and model card (the artifact triple)
schema = {
    "features": sorted(X_val.columns.tolist()),
    "numeric": sorted([
        "age", "campaign", "pdays", "previous",
        "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed", "pdays_is_sentinel"
    ]),
    "categorical": sorted([
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "poutcome"
    ]),
}
data_hash = hashlib.sha256(open(VAL_PATH, "rb").read()).hexdigest()
fingerprint = {
    "python_version": py_platform.python_version(),
    "sklearn_version": sklearn_version,
    "pandas_version": pd.__version__,
}
model_card = (
    f"# Model Card\n"
    f"**Model:** {REGISTERED_MODEL_NAME}\n"
    f"**Val AUC:** {val_auc:.4f}\n"
    f"**Val Recall:** {val_recall:.4f}\n"
    f"**Operating Threshold:** {threshold_data['threshold']:.6f}\n"
    f"**Val Data Hash (sha256):** {data_hash}\n"
    f"**Environment:** {json.dumps(fingerprint)}\n"
)

# One-step: log model + artifacts + real metrics, then register
with mlflow.start_run(run_name="bank_marketing_model") as run:
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    mlflow.log_dict(schema, "schema.json")
    mlflow.log_text(model_card, "model_card.md")
    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_metric("val_recall", val_recall)
    mlflow.log_metric("operating_threshold", float(threshold_data["threshold"]))
    run_id = run.info.run_id

# Find the latest version (the one just created)
versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
latest = max(versions, key=lambda v: int(v.version))

# Promote to Production
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=latest.version,
    stage="Production",
    archive_existing_versions=True
)
print(f"Model version {latest.version} registered and promoted to Production.")