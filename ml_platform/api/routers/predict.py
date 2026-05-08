import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException

from ml_platform.config import *
from ml_platform.drift.compute import RollingWindow, psi, chi2, output_drift, compute_severity
from ml_platform.schemas import PredictionRequest, PredictionResponse


# ------------------------------------------------------------
# Logging – make sure webhook messages are visible
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

DRIFT_STATE_FILE = Path("/app/data/drift_state.json")

def _load_drift_state():
    """Return (last_severity, last_success_time) from disk, or defaults."""
    try:
        data = json.loads(DRIFT_STATE_FILE.read_text())
        return data.get("last_severity", "green"), data.get("last_success_time", 0.0)
    except Exception:
        return "green", 0.0

def _save_drift_state(severity, success_time):
    """Persist severity and last successful webhook time."""
    DRIFT_STATE_FILE.write_text(json.dumps({
        "last_severity": severity,
        "last_success_time": success_time
    }))


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
_reference_pos_rate = None
_last_severity, _last_webhook_time = _load_drift_state()

def get_reference_stats():
    global _reference_stats
    if _reference_stats is None:
        with open(REFERENCE_STATS_PATH) as f:
            _reference_stats = json.load(f)
    return _reference_stats

def get_reference_pos_rate():
    """Model's predicted positive rate on training data — the right baseline for
    output drift. Comparing current_pos_rate to the *label* rate would always
    show a huge gap whenever the threshold is recall-tuned (this model: ~0.40
    predicted vs ~0.11 labels), making the drift gate fire on healthy traffic."""
    global _reference_pos_rate
    if _reference_pos_rate is None:
        import joblib
        df = pd.read_parquet(TRAIN_PATH).drop(columns=[TARGET_COL])
        pipeline = joblib.load(PIPELINE_PATH)
        with open(THRESHOLD_PATH) as f:
            thr = json.load(f)["threshold"]
        probs = pipeline.predict_proba(df)[:, 1]
        _reference_pos_rate = float((probs >= thr).mean())
    return _reference_pos_rate

def to_json_safe(obj):
    """Recursively convert sets → lists so the payload is JSON‑serializable."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    return obj

async def emit_webhook(severity: str, prev_severity: str, drift_metrics: dict, model_version: str = "") -> bool:
    now = datetime.now(timezone.utc)

    payload = {
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "event_id": str(uuid.uuid4()),
        "model_id": REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "severity": severity,
        "previous_severity": prev_severity,
        "current_window": {
            "start": (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            "end": now.isoformat().replace("+00:00", "Z"),
            "num_predictions": len(_window.records)
        },
        "psi": drift_metrics["psi"],
        "chi2": drift_metrics["chi2"],
        "output_drift": drift_metrics["output_drift"]
    }

    body = json.dumps(payload).encode("utf-8")

    signature = hmac.new(
        AGENT_WEBHOOK_SECRET.encode("utf-8"),
        body,
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-Contract-Version": "drift-alert-v1",
        "X-HMAC-Signature": signature,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                AGENT_DRIFT_WEBHOOK_URL,
                content=body,
                headers=headers,
                timeout=5
            )
            logger.info(f"Webhook sent, status {resp.status_code}")
            # consider 200/202 as success, others as failure
            return resp.status_code in (200, 202)
        except Exception as e:
            logger.error(f"Webhook failed: {e}")
            return False

def compute_drift():
    if len(_window.records) < 30:
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
    od = output_drift(current_pos_rate, get_reference_pos_rate())

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
    global _last_severity, _last_webhook_time

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

        send_reason = None
        if severity != _last_severity:
            send_reason = "transition"
        elif severity != "green" and (time.time() - _last_webhook_time > 60):
            # Re-alert if non-green severity has persisted >60s (demo helper)
            send_reason = "heartbeat"

        if send_reason is not None:
            success = await emit_webhook(severity, _last_severity, drift_metrics, model_version=str(_version))
            if success:
                _last_severity = severity
                _last_webhook_time = time.time()
                _save_drift_state(severity, _last_webhook_time)
                logger.info(f"Webhook delivered (reason: {send_reason})")
            else:
                logger.warning(f"Webhook delivery failed, will retry (reason: {send_reason})")

    return PredictionResponse(probability=prob, prediction=pred)