import pandas as pd
from sklearn.model_selection import train_test_split
import json
from ml_platform.config import *

def load_raw():
    return pd.read_csv(DATA_PATH, sep=";")

def clean(df):
    df = df.copy()
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
    df["pdays_is_sentinel"] = (df[SENTINEL_COL] == SENTINEL_VALUE).astype(int)
    df[SENTINEL_COL] = df[SENTINEL_COL].replace(SENTINEL_VALUE, -1)
    df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0})
    return df

def split_data(df):
    train, temp = train_test_split(
        df, test_size=0.4, stratify=df[TARGET_COL], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp[TARGET_COL], random_state=RANDOM_STATE
    )
    return train, val, test

def compute_reference_stats(train_df):
    stats = {}
    numeric_cols = NUMERIC_COLS + ["pdays_is_sentinel"]
    for col in numeric_cols:
        stats[f"{col}_min"] = float(train_df[col].min())
        stats[f"{col}_max"] = float(train_df[col].max())
        stats[f"{col}_mean"] = float(train_df[col].mean())
        stats[f"{col}_std"] = float(train_df[col].std())
        out = pd.cut(train_df[col], bins=10, retbins=True)
        _, bins = out[0], out[1]
        counts = out[0].value_counts(normalize=True).sort_index()
        stats[f"{col}_bins"] = bins.tolist()
        stats[f"{col}_bin_props"] = counts.values.tolist()

    for col in CATEGORICAL_COLS:
        freq_counts = train_df[col].value_counts(normalize=True).to_dict()
        stats[f"{col}_freqs"] = freq_counts

    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    return stats

if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    train, val, test = split_data(df)
    train.to_parquet(TRAIN_PATH, index=False)
    val.to_parquet(VAL_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)
    compute_reference_stats(train)
    print("✅ Preprocessing complete – splits and stats saved.")