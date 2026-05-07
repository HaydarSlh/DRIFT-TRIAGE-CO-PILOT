import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db')
client = MlflowClient()

versions = client.search_model_versions('name=BankMarketingClassifier')
latest = max(versions, key=lambda v: int(v.version))

client.transition_model_version_stage(
    name='BankMarketingClassifier',
    version=latest.version,
    stage='Production',
    archive_existing_versions=True
)
print(f'Promoted version {latest.version} to Production.')