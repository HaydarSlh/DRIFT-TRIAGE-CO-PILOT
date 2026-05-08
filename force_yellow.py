"""Send drifted rows to /predict to trigger a YELLOW severity webhook.

Three platform signals must all stay in the yellow band (none in red):
  - PSI on each NUMERIC col + pdays_is_sentinel (yellow: 0.10 ≤ max < 0.25)
  - chi2 on each CATEGORICAL col (we don't shift these, so they stay green)
  - |output_drift| (yellow has no contribution; just keep |od| ≤ 0.10)

The simulator runs the SAME transform predict.py runs (pdays==999 → sentinel)
and the SAME model the platform serves (pipeline.pkl), so simulated severity
matches what the platform will compute once these rows hit its rolling window.

Note: train.parquet stores pdays already replaced with -1 for sentinels, but
the platform expects raw 999 on the wire. We restore that before sending.
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
)
from ml_platform.drift.compute import psi, compute_severity

TARGET_COL_TO_SHIFT = "age"
N_ROWS = 500
PREDICT_URL = "http://localhost:8080/predict"
REF_STATS_PATH = "artifacts/reference_stats.json"
PIPELINE_PATH = "artifacts/pipeline.pkl"
THRESHOLD_PATH = "artifacts/threshold.json"

SHIFT_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# How many of the N_ROWS get the shift. Smaller fraction dampens both PSI
# and output_drift, useful when full-shift overshoots one of them.
DRIFT_FRACTION_SWEEP = [1.0, 0.7, 0.5, 0.3]


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
    """Reverse the pdays==999 → -1 transform, so predict.py sees raw 999."""
    df = df.copy()
    df.loc[df["pdays"] == -1, "pdays"] = 999
    return df


def apply_predict_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror predict.py: derive pdays_is_sentinel, replace 999 with -1."""
    df = df.copy()
    df["pdays_is_sentinel"] = (df["pdays"] == 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)
    return df


def simulate(rows_raw: pd.DataFrame, pipeline, threshold: float, ref_stats: dict, ref_pos_rate: float):
    """Simulate the platform's drift computation on rows_raw (pre-transform)."""
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

    current_pos_rate = float(preds.mean())
    od = current_pos_rate - ref_pos_rate
    severity = compute_severity(psi_vals, {}, output_drift_val=od)
    return severity, psi_vals, od


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH).drop(columns=[TARGET_COL])

    with open(REF_STATS_PATH) as f:
        ref_stats = json.load(f)
    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]
    pipeline = joblib.load(PIPELINE_PATH)

    # Match predict.py:get_reference_pos_rate — baseline is the model's predicted
    # positive rate on training, NOT the label rate, so the threshold's recall
    # tuning doesn't produce permanent baseline drift.
    ref_probs = pipeline.predict_proba(df)[:, 1]
    ref_pos_rate = float((ref_probs >= threshold).mean())

    base = df.sample(n=N_ROWS, random_state=42).reset_index(drop=True)
    base = restore_sentinel(base)
    std = float(df[TARGET_COL_TO_SHIFT].std())

    sev, psi_vals, od = simulate(base, pipeline, threshold, ref_stats, ref_pos_rate)
    max_col = max(psi_vals, key=psi_vals.get)
    print(f"Reference pos rate: {ref_pos_rate:.3f}")
    print(f"Baseline (no shift): {sev}  max PSI {max_col}={psi_vals[max_col]:.3f}  output_drift={od:+.3f}")
    print()

    print(f"Sweeping shifts on '{TARGET_COL_TO_SHIFT}' × drift fraction:")
    print(f"  yellow band: 0.10 ≤ max_psi < 0.25  AND  |output_drift| ≤ 0.10")
    chosen = None
    for frac in DRIFT_FRACTION_SWEEP:
        for coef in SHIFT_SWEEP:
            rows = base.copy()
            n_shifted = int(N_ROWS * frac)
            shifted = rows[TARGET_COL_TO_SHIFT].iloc[:n_shifted] + coef * std
            if pd.api.types.is_integer_dtype(rows[TARGET_COL_TO_SHIFT].dtype):
                shifted = shifted.round().astype(rows[TARGET_COL_TO_SHIFT].dtype)
            rows.loc[rows.index[:n_shifted], TARGET_COL_TO_SHIFT] = shifted

            sev, psi_vals, od = simulate(rows, pipeline, threshold, ref_stats, ref_pos_rate)
            max_col = max(psi_vals, key=psi_vals.get)
            marker = ""
            if sev == "yellow" and chosen is None:
                chosen = (frac, coef, rows)
                marker = "  ← picked"
            print(
                f"  frac={frac:.1f}  +{coef:.2f}*std: "
                f"max PSI {max_col}={psi_vals[max_col]:.3f}  od={od:+.3f}  → {sev}{marker}"
            )

    if chosen is None:
        print("\nNo (frac, shift) combo produced yellow. Widen the sweeps.")
        return

    frac, coef, chosen_rows = chosen
    print(
        f"\n✅ Sending {N_ROWS} rows: {int(frac * 100)}% with +{coef:.2f}*std on "
        f"'{TARGET_COL_TO_SHIFT}', rest unshifted, to {PREDICT_URL}"
    )
    sent = 0
    for i, (_, r) in enumerate(chosen_rows.iterrows(), start=1):
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
    print(f"\nDone. Sent {sent}/{N_ROWS}. Check agent logs / dashboard for the yellow webhook.")


if __name__ == "__main__":
    main()
