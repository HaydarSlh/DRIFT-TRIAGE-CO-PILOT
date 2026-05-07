import mlflow
from mlflow.tracking import MlflowClient
from ml_platform.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

def promote_to_production(model_name, version):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Promote the given version to Production, archive any existing Production version
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

    # Verify the promotion
    mv = client.get_model_version(model_name, version)
    print(f"Model '{model_name}' version {mv.version} is now in stage: {mv.current_stage}")

if __name__ == "__main__":
    # Default: promote version 1 (change if needed)
    promote_to_production(REGISTERED_MODEL_NAME, 1)