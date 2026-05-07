import uuid
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import json
import mlflow
from datetime import datetime, timezone
from ml_platform.config import *
from ml_platform.drift.compute import RollingWindow, psi, chi2, output_drift, compute_severity
from ml_platform.schemas import PredictionRequest, PredictionResponse, DriftAlert, DriftAlertWindow, DriftAlertDetails
import httpx

# ------------------------------------------------------------
# Logging – make sure webhook messages are visible
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------
# Model cache
# ------------------------------------------------------------
_model = None
_threshold = None
_version = None

def get_model():
    global _model, _threshold, _version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        # Non‑deprecated search for Production version
        filter_string = f"name='{REGISTERED_MODEL_NAME}'"
        results = client.search_model_versions(filter_string)
        prod = [mv for mv in results if mv.current_stage == "Production"]
        if not prod:
            raise ValueError("No production model found")
        latest = sorted(prod, key=lambda x: int(x.version))[-1]
        if _version != latest.version:
            _model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{latest.version}")
            with open(THRESHOLD_PATH) as f:
                _threshold = json.load(f)["threshold"]
            _version = latest.version
    except Exception as e:
        logger.error(f"Model load error: {e}")
        raise HTTPException(status_code=503, detail="Model unavailable")
    return _model, _threshold

# ------------------------------------------------------------
# Drift state
# ------------------------------------------------------------
_window = RollingWindow()
_reference_stats = None
_last_severity = "green"

def get_reference_stats():
    global _reference_stats
    if _reference_stats is None:
        with open(REFERENCE_STATS_PATH) as f:
            _reference_stats = json.load(f)
    return _reference_stats

def to_json_safe(obj):
    """Recursively convert sets → lists so the payload is JSON‑serializable."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    return obj

async def emit_webhook(severity, prev_severity, drift_metrics):
    payload = DriftAlert(
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        event_id=str(uuid.uuid4()),
        severity=severity,
        previous_severity=prev_severity,
        current_window=DriftAlertWindow(
            start="TODO",
            end=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            num_predictions=len(_window.records)
        ),
        drift_details=DriftAlertDetails(
            psi=drift_metrics["psi"],
            chi2=drift_metrics["chi2"],
            output_drift=drift_metrics["output_drift"]
        )
    )
    headers = {"X-Contract-Version": "drift-alert-v1"}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(AGENT_DRIFT_WEBHOOK_URL, json=payload.model_dump(), headers=headers, timeout=5)
            logger.info(f"Webhook sent, status {resp.status_code}")
        except Exception as e:
            logger.error(f"Webhook failed: {e}")

def compute_drift():
    if len(_window.records) < 50:
        return None
    df = _window.to_df()
    stats = get_reference_stats()
    psi_vals = {}
    for col in NUMERIC_COLS + ["pdays_is_sentinel"]:
        if col not in df.columns:
            continue
        bins = stats[f"{col}_bins"]
        props = stats[f"{col}_bin_props"]
        psi_vals[col] = psi(df[col].values, bins, props)

    chi2_vals = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        freqs = stats[f"{col}_freqs"]
        chi2_vals[col] = chi2(df[col], freqs)

    if "prediction" in df.columns:
        current_pos_rate = (df["prediction"] == 1).mean()
    else:
        current_pos_rate = 0.0
    ref_pos_rate = float(pd.read_parquet(TRAIN_PATH)[TARGET_COL].mean())
    od = output_drift(current_pos_rate, ref_pos_rate)

    return {
        "psi": psi_vals,
        "chi2": chi2_vals,
        "output_drift": od
    }

# ------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global _last_severity

    # 1. Load model
    try:
        model, threshold = get_model()
    except Exception:
        raise HTTPException(status_code=503, detail="Model not available")

    # 2. Convert request to DataFrame (use aliases so columns match the training names)
    input_data = pd.DataFrame([request.model_dump(by_alias=True)])

    # 3. Add pdays_is_sentinel
    input_data["pdays_is_sentinel"] = (input_data["pdays"] == 999).astype(int)
    input_data["pdays"] = input_data["pdays"].replace(999, -1)

    # 4. Predict
    prob = model.predict_proba(input_data)[:, 1][0]
    pred = int(prob >= threshold)

    # 5. Log for drift
    record = input_data.iloc[0].to_dict()
    record["probability"] = prob
    record["prediction"] = pred
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    record["model_version"] = _version
    _window.add(record)

    # 6. Drift check
    drift_metrics = compute_drift()
    if drift_metrics:
        severity = compute_severity(
            drift_metrics["psi"],
            drift_metrics["chi2"],
            drift_metrics["output_drift"]
        )
        if severity != _last_severity:
            await emit_webhook(severity, _last_severity, drift_metrics)
            _last_severity = severity

    return PredictionResponse(probability=prob, prediction=pred)