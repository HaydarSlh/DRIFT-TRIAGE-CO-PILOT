import requests
import pandas as pd
import numpy as np
from ml_platform.config import TRAIN_PATH, TARGET_COL

df = pd.read_parquet(TRAIN_PATH).drop(columns=[TARGET_COL])
std_euribor = df['euribor3m'].std()

print("Sending 600 drifted rows...")
for i in range(100):
    row = df.iloc[i % len(df)].to_dict()
    row['euribor3m'] = float(row['euribor3m']) + 2 * std_euribor
    row['job'] = 'housemaid'
    # convert numpy types to native Python
    row = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in row.items()}
    resp = requests.post("http://localhost:8080/predict", json=row)
    if i % 100 == 0:
        print(f"Sent {i} drifted rows...")

print("Done. Check server console for 'Webhook sent'.")