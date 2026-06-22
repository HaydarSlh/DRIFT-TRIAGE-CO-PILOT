"""Send drifted rows that bias the agent toward `retrain_shadow`.

The agent's action is LLM-graded ({none, replay_test, retrain_shadow, rollback}).
There's no threshold that guarantees retrain. We bias the choice by shaping
the drift to *look* like it should be fixable by retraining:

  - Strong numeric distribution shift on age and previous   → "model needs
    to relearn the new feature distribution"
  - No categorical breakage (job left alone)                → not "model is broken"
  - Output drift moderate (not extreme)                     → not rollback-tier

Triage prompt buckets HIGH as "significant drift, likely needs retraining or
rollback." Without categorical breakage, the LLM tends to pick retrain.

After running this, the agent will pause on an HIL interrupt for approval.
Open the dashboard's pending-approvals page and click approve — only then
does /retrain actually fire.
"""
import json

import joblib
import numpy as np
import pandas as pd
import requests

from ml_platform.config import (
    TRAIN_PATH,
    TARGET_COL,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
)
from ml_platform.drift.compute import psi, chi2, compute_severity

N_ROWS = 500
PREDICT_URL = "http://localhost:8080/predict"
REF_STATS_PATH = "artifacts/reference_stats.json"
PIPELINE_PATH = "artifacts/pipeline.pkl"
THRESHOLD_PATH = "artifacts/threshold.json"

# Significant numeric drift, no categorical drift, moderate output drift.
# Tuned to land HIGH (not CRITICAL) with retrain-shaped signals.
NUMERIC_SHIFTS = {"age": 0.7, "previous": 0.7}


def to_native(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def restore_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["pdays"] == -1, "pdays"] = 999
    return df


def apply_predict_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pdays_is_sentinel"] = (df["pdays"] == 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)
    return df


def simulate(rows_raw: pd.DataFrame, pipeline, threshold: float, ref_stats: dict, ref_pos_rate: float):
    df = apply_predict_transform(rows_raw)
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)

    psi_vals = {}
    for col in NUMERIC_COLS + ["pdays_is_sentinel"]:
        if col not in df.columns:
            continue
        psi_vals[col] = psi(
            df[col].values,
            ref_stats[f"{col}_bins"],
            ref_stats[f"{col}_bin_props"],
        )

    chi2_vals = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        chi2_vals[col] = chi2(df[col], ref_stats[f"{col}_freqs"])

    od = float(preds.mean()) - ref_pos_rate
    severity = compute_severity(psi_vals, chi2_vals, output_drift_val=od)
    return severity, psi_vals, chi2_vals, od


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH).drop(columns=[TARGET_COL])

    with open(REF_STATS_PATH) as f:
        ref_stats = json.load(f)
    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]
    pipeline = joblib.load(PIPELINE_PATH)

    ref_probs = pipeline.predict_proba(df)[:, 1]
    ref_pos_rate = float((ref_probs >= threshold).mean())

    base = df.sample(n=N_ROWS, random_state=42).reset_index(drop=True)
    base = restore_sentinel(base)

    rows = base.copy()
    for col, coef in NUMERIC_SHIFTS.items():
        std = float(df[col].std())
        shifted = rows[col] + coef * std
        if pd.api.types.is_integer_dtype(rows[col].dtype):
            shifted = shifted.round().astype(rows[col].dtype)
        rows[col] = shifted

    sev, psi_vals, chi2_vals, od = simulate(rows, pipeline, threshold, ref_stats, ref_pos_rate)
    max_psi_col = max(psi_vals, key=psi_vals.get)
    max_chi2_col = max(chi2_vals, key=chi2_vals.get) if chi2_vals else None

    print(f"Reference pos rate: {ref_pos_rate:.3f}")
    print(f"Simulated drift signals on {N_ROWS} rows:")
    print(f"  numeric shifts: {NUMERIC_SHIFTS}")
    print(f"  no categorical drift")
    print(f"  max PSI: {max_psi_col}={psi_vals[max_psi_col]:.3f}")
    if max_chi2_col:
        print(f"  max chi²: {max_chi2_col}={chi2_vals[max_chi2_col]:.1f}")
    print(f"  output_drift: {od:+.3f}")
    print(f"  platform severity: {sev}")
    print(f"  → triage should classify HIGH; action should pick retrain_shadow.")
    print()

    if sev == "green":
        print("⚠️  Platform sees green — agent won't act. Increase shifts and rerun.")
        return

    print(f"✅ Sending {N_ROWS} rows to {PREDICT_URL}")
    sent = 0
    for i, (_, r) in enumerate(rows.iterrows(), start=1):
        try:
            resp = requests.post(PREDICT_URL, json=to_native(r.to_dict()), timeout=5)
            if resp.status_code == 200:
                sent += 1
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            print(f"  request failed: {e}")
        if i % 100 == 0:
            print(f"  sent {i}/{N_ROWS}")
    print(f"\nDone. Sent {sent}/{N_ROWS}.")
    print("Now: open the dashboard's pending-approvals page and approve the retrain request.")
    print("Only after approval will the worker call POST /retrain on the platform.")


if __name__ == "__main__":
    main()
