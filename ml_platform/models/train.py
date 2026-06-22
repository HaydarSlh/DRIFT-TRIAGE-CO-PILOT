import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
import json
import joblib
import numpy as np
from ml_platform.config import *
import os
custom_train = os.getenv("TRAIN_DATA_PATH")
if custom_train:
    TRAIN_PATH = Path(custom_train)

def build_pipeline():
    numeric_features = NUMERIC_COLS + ["pdays_is_sentinel"]
    categorical_features = CATEGORICAL_COLS

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ])

    # Classifier with balanced class weights
    classifier = RandomForestClassifier(
        class_weight='balanced',   # ✅ automatically adjust for imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline

if __name__ == "__main__":
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    test = pd.read_parquet(TEST_PATH)

    X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
    X_val, y_val = val.drop(columns=[TARGET_COL]), val[TARGET_COL]
    X_test, y_test = test.drop(columns=[TARGET_COL]), test[TARGET_COL]

    pipeline = build_pipeline()

    # ------------------------------------------------------------
    # Hyperparameter tuning with RandomizedSearchCV
    # ------------------------------------------------------------
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,                          # try 20 random combos
        scoring='roc_auc',                  # optimize for AUC
        cv=3,                               # 3‑fold cross‑validation
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    print("Best params:", search.best_params_)
    print("Best CV score (AUC):", search.best_score_)

    # ------------------------------------------------------------
    # Threshold tuning on validation set
    # ------------------------------------------------------------
    y_prob_val = best_pipeline.predict_proba(X_val)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_val, y_prob_val)

    meets_recall = rec[:-1] >= 0.75
    if meets_recall.any():
        idx = meets_recall.nonzero()[0][-1]      # last True → highest threshold
        op_threshold = float(thresh[idx])
    else:
        op_threshold = 0.5
        print("No threshold meets recall >= 0.75 - using default 0.5")

    val_preds = (y_prob_val >= op_threshold).astype(int)
    val_recall = recall_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, y_prob_val)
    print(f"Threshold {op_threshold:.4f} | Val Recall {val_recall:.3f} | "
          f"F1 {val_f1:.3f} | AUC {val_auc:.3f}")
    
    
    
    import mlflow
    from ml_platform.config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="training_metrics_run") as run:
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("val_recall", val_recall)

    # ------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------
    y_prob_test = best_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (y_prob_test >= op_threshold).astype(int)
    test_auc = roc_auc_score(y_test, y_prob_test)
    test_f1 = f1_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    print(f"Test AUC {test_auc:.3f} | F1 {test_f1:.3f} | Recall {test_recall:.3f}")

    # Save
    joblib.dump(best_pipeline, PIPELINE_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": op_threshold, "val_recall": float(val_recall)}, f)
    print(f"Pipeline saved to {PIPELINE_PATH}")