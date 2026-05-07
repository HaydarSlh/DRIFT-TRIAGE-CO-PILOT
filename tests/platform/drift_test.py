import sys
from pathlib import Path

# Add the project root to sys.path so we can import ml_platform
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests
import pandas as pd
import numpy as np
from ml_platform.config import TRAIN_PATH, TARGET_COL
BASE_URL = "http://localhost:8000/predict"

# Load a batch of test data (you could also use train)
df = pd.read_parquet(TRAIN_PATH).drop(columns=[TARGET_COL])

# ---- Step A: Send 200 normal rows to fill the rolling window ----
print("Sending 200 normal rows...")
for i in range(200):
    row = df.iloc[i].to_dict()
    # Convert numpy types
    row = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in row.items()}
    resp = requests.post(BASE_URL, json=row)
print("Done. Window should be full. Severity: green (probably).")

# ---- Step B: Send shifted rows to trigger drift ----
print("Sending 200 drifted rows...")
# Alter a numeric feature: increase euribor3m by 2 standard deviations
std_euribor = df['euribor3m'].std()
for i in range(200, 400):
    row = df.iloc[i].to_dict()
    # shift euribor3m
    row['euribor3m'] = float(row['euribor3m']) + 2 * std_euribor
    # shift a categorical: set job to a rare value like 'student' (if it exists) or something unusual
    # (the dataset has categories like 'admin.', 'blue-collar', etc. – we'll just set it to 'housemaid')
    row['job'] = 'housemaid'   # this may not exist in training, so it's a strong shift
    row = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in row.items()}
    resp = requests.post(BASE_URL, json=row)
    if i % 50 == 0:
        print(f"Sent {i-200} drifted rows...")

print("Done. Check the server console for 'Webhook sent'.")