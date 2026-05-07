import pandas as pd
import numpy as np
from ml_platform.config import TRAIN_PATH, TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS

def create_drifted_train(output_path=None):
    # Load the original 60% train split
    train = pd.read_parquet(TRAIN_PATH)
    df = train.copy()

    # Shift euribor3m by +2 standard deviations (same as your demo drift)
    std_euribor = df['euribor3m'].std()
    df['euribor3m'] = df['euribor3m'] + 2 * std_euribor

    # Shift categorical: set 'job' to 'housemaid' for half the rows (creates strong shift)
    rng = np.random.default_rng(42)
    mask = rng.choice(df.index, size=len(df)//2, replace=False)
    df.loc[mask, 'job'] = 'housemaid'

    # Optionally also shift cons.price.idx, but one numeric + one categorical is enough

    # Save as a new file (do not overwrite original)
    if output_path is None:
        output_path = TRAIN_PATH.parent / "drifted_train.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Drifted training data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    create_drifted_train()