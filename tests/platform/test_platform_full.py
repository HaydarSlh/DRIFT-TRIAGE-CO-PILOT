import sys
from pathlib import Path

# Add the project root to sys.path so we can import ml_platform
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path


# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_platform.config import (
    TRAIN_PATH, TEST_PATH, TARGET_COL,
    MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME,
    THRESHOLD_PATH
)

BASE = "http://localhost:8000"
AGENT = "http://agent:8001/webhooks/drift"   # may be down

# -----------------------------------------------
# Helper to convert numpy types to native Python
# -----------------------------------------------
def py_row(row):
    return {k: (int(v) if isinstance(v, (np.integer,))
                else float(v) if isinstance(v, (np.floating,))
                else v)
            for k, v in row.items()}

# -----------------------------------------------
# 1. Health check
# -----------------------------------------------
def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    print("✅ Health check passed")

# -----------------------------------------------
# 2. Valid prediction
# -----------------------------------------------
def test_valid_prediction():
    train = pd.read_parquet(TRAIN_PATH)
    row = py_row(train.drop(columns=[TARGET_COL]).iloc[0])
    r = requests.post(f"{BASE}/predict", json=row)
    assert r.status_code == 200
    data = r.json()
    assert "probability" in data
    assert "prediction" in data
    print(f"✅ Valid prediction: probability={data['probability']:.4f}, prediction={data['prediction']}")

# -----------------------------------------------
# 3. Validation errors
# -----------------------------------------------
def test_missing_field():
    r = requests.post(f"{BASE}/predict", json={"age": 30})
    assert r.status_code == 422
    print("✅ Missing field → 422")

def test_bad_json():
    r = requests.post(f"{BASE}/predict", data="not json",
                      headers={"Content-Type": "application/json"})
    assert r.status_code == 400 or r.status_code == 422
    print("✅ Bad JSON → 4xx")

def test_wrong_type():
    row = {"age": "thirty", "job": "admin.", "marital": "married",
           "education": "university.degree", "default": "no", "housing": "yes",
           "loan": "no", "contact": "cellular", "month": "may",
           "day_of_week": "mon", "campaign": 1, "pdays": 999, "previous": 0,
           "poutcome": "nonexistent",
           "emp.var.rate": 1.1, "cons.price.idx": 93.994,
           "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0}
    r = requests.post(f"{BASE}/predict", json=row)
    assert r.status_code == 422
    print("✅ Wrong type → 422")

# -----------------------------------------------
# 4. Drift webhook (green → red transition)
# -----------------------------------------------
def test_drift_webhook():
    import subprocess
    import time

    # You must restart the server MANUALLY before this test so the rolling window is clean.
    print("⚠️  Please restart the uvicorn server now (Ctrl+C and start again). Press Enter to continue...")
    input()

    train = pd.read_parquet(TRAIN_PATH).drop(columns=[TARGET_COL])
    std_euribor = train['euribor3m'].std()

    # Phase A: fill window with 200 normal rows
    print("Sending 200 normal rows...")
    for i in range(200):
        row = py_row(train.iloc[i])
        requests.post(f"{BASE}/predict", json=row)

    # Phase B: inject drift (increase euribor3m, change job)
    print("Injecting drift (shifting euribor3m & job)...")
    for i in range(200, 500):
        row = py_row(train.iloc[i % len(train)])
        row['euribor3m'] = float(row['euribor3m']) + 2 * std_euribor
        row['job'] = 'housemaid'
        requests.post(f"{BASE}/predict", json=row)

    # Check server log manually for "Webhook sent" or "Webhook failed" (agent may be down)
    print("Check the uvicorn console. You should see something like:")
    print("   'Webhook sent, status 202'   OR   'Webhook failed: ...'")
    print("If you see it, drift webhook works. ✅")

# -----------------------------------------------
# 5. Promotion endpoint tests
# -----------------------------------------------
def test_promotion_correct():
    from datetime import datetime, timezone

    headers = {"X-Contract-Version": "promotion-v1"}
    now = datetime.now(timezone.utc)
    ts = now.isoformat()   # e.g., "2026-05-07T12:34:56.789123+00:00"
    # Replace +00:00 with Z for the contract
    ts_z = ts.replace("+00:00", "Z")

    payload = {
        "action": "promote",
        "model_version": "1",
        "investigation_id": "test-id",
        "approved_by": "tester",
        "approval_timestamp": ts_z,
        "reason": "testing promotion gate"
    }
    r = requests.post(f"{BASE}/registry/promote", json=payload, headers=headers)
    print(f"Promotion response ({r.status_code}): {r.json()}")
    assert r.status_code in [200, 409]
    print("✅ Promotion endpoint reached correctly")

def test_promotion_missing_header():
    payload = {"action": "promote", "model_version": "1", "investigation_id": "x",
               "approved_by": "tester", "approval_timestamp": "2026-05-07T00:00:00Z",
               "reason": "no header"}
    r = requests.post(f"{BASE}/registry/promote", json=payload)
    assert r.status_code == 400
    print("✅ Missing contract header → 400")

def test_promotion_expired():
    from datetime import datetime, timezone, timedelta

    headers = {"X-Contract-Version": "promotion-v1"}
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    ts_z = old_time.isoformat().replace("+00:00", "Z")

    payload = {
        "action": "promote",
        "model_version": "1",
        "investigation_id": "x",
        "approved_by": "tester",
        "approval_timestamp": ts_z,
        "reason": "expired approval"
    }
    r = requests.post(f"{BASE}/registry/promote", json=payload, headers=headers)
    print(f"Expired approval response ({r.status_code}): {r.json()}")
    assert r.status_code == 410
    print("✅ Expired approval → 410")

# -----------------------------------------------
# 6. Model fidelity test (run separately)
# -----------------------------------------------

# -----------------------------------------------
# Run all tests
# -----------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    test_health()
    test_valid_prediction()
    test_missing_field()
    test_bad_json()
    test_wrong_type()
    test_promotion_correct()
    test_promotion_missing_header()
    test_promotion_expired()


    # Drift test requires manual server restart, so we put it last
    test_drift_webhook()

    print("\n🎉 All automated tests passed. Verify the drift webhook manually.")