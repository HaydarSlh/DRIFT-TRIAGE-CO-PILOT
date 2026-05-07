import mlflow
import mlflow.sklearn
import joblib
import json
import pandas as pd
import hashlib
import platform as py_platform
from sklearn import __version__ as sklearn_version
from ml_platform.config import *

def make_schema(train_df):
    features = sorted(list(train_df.drop(columns=[TARGET_COL]).columns))
    return {
        "features": features,
        "numeric": sorted(NUMERIC_COLS + ["pdays_is_sentinel"]),
        "categorical": sorted(CATEGORICAL_COLS),
    }

def make_model_card(run_id, fingerprint):
    data_hash = hashlib.sha256(open(TRAIN_PATH, 'rb').read()).hexdigest()
    return f"""# Model Card
**Run ID:** {run_id}
**Environment Fingerprint:** {fingerprint}
**Data Hash:** {data_hash}
"""

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    pipeline = joblib.load(PIPELINE_PATH)
    with open(THRESHOLD_PATH) as f:
        threshold_data = json.load(f)

    train_df = pd.read_parquet(TRAIN_PATH)
    schema = make_schema(train_df)

    fingerprint = {
        "python_version": py_platform.python_version(),
        "sklearn_version": sklearn_version,
        "pandas_version": pd.__version__,
    }
    fingerprint_str = json.dumps(fingerprint)

    # Log the model AND register it in one call
    with mlflow.start_run(run_name="bank_marketing_model") as run:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",          # directory inside the run where the model will live
            registered_model_name=REGISTERED_MODEL_NAME,   # Auto-register after logging
            signature=None,                 # optional – skip for now
        )
        mlflow.log_dict(schema, "schema.json")
        card = make_model_card(run.info.run_id, fingerprint_str)
        mlflow.log_text(card, "model_card.md")
        mlflow.log_param("operating_threshold", threshold_data["threshold"])
        mlflow.log_metric("val_recall", threshold_data["val_recall"])

    print(f"Model registered as {REGISTERED_MODEL_NAME} latest version")