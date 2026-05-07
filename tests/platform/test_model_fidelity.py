import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from ml_platform.config import TEST_PATH, PIPELINE_PATH, TARGET_COL

SNAPSHOT_PATH = Path(__file__).resolve().parent / "test_probs_snapshot.npy"

def test_model_fidelity():
    # Load pipeline
    pipeline = joblib.load(PIPELINE_PATH)

    # Load test data
    test = pd.read_parquet(TEST_PATH)
    X_test = test.drop(columns=[TARGET_COL])

    # Predict probabilities
    probs = pipeline.predict_proba(X_test)[:, 1]

    # Snapshot management
    if not SNAPSHOT_PATH.exists():
        # First run: save the snapshot
        np.save(SNAPSHOT_PATH, probs)
        print(f"Snapshot saved to {SNAPSHOT_PATH}")
        return  # in CI you'd skip assertion on creation, or better: raise a warning
    else:
        # Subsequent runs: compare
        saved = np.load(SNAPSHOT_PATH)
        max_diff = np.max(np.abs(probs - saved))
        assert max_diff < 1e-12, f"Model fidelity broken: max diff = {max_diff:.2e}"
        print(f"Fidelity test passed. Max diff = {max_diff:.2e}")

if __name__ == "__main__":
    test_model_fidelity()