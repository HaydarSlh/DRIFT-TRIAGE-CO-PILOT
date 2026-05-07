from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
TRAIN_PATH = ROOT / "data" / "train.parquet"
VAL_PATH = ROOT / "data" / "val.parquet"
TEST_PATH = ROOT / "data" / "test.parquet"

# Artifacts
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
PIPELINE_PATH = ARTIFACT_DIR / "pipeline.pkl"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.json"
REFERENCE_STATS_PATH = ARTIFACT_DIR / "reference_stats.json"

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
REGISTERED_MODEL_NAME = "BankMarketingClassifier"

RANDOM_STATE = 42
TARGET_COL = "y"
DROP_COLS = ["duration"]
SENTINEL_COL = "pdays"
SENTINEL_VALUE = 999

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome"
]

NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]

AGENT_DRIFT_WEBHOOK_URL = "http://agent:8001/webhooks/drift"