@echo off
call python -m ml_platform.drifted_data
set TRAIN_DATA_PATH=data/drifted_train.parquet
call python -m ml_platform.models.train
call python mlflow/register.py