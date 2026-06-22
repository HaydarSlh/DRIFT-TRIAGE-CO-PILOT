FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ml_platform/requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

# Copy the entire ml_platform package
COPY ml_platform/ ml_platform/

# Copy data & artifacts
COPY data/ data/
COPY artifacts/ artifacts/

RUN mkdir -p /app/mlruns

EXPOSE 8000 5000

# Register the pre-trained model to Production, then start servers
CMD ["sh", "-c", "\
  python scripts/final_register.py && \
  mlflow server \
    --backend-store-uri sqlite:////app/mlruns/mlflow.db \
    --default-artifact-root /app/mlruns/artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts '*' \
    --cors-allowed-origins '*' & \
  uvicorn ml_platform.api.main:app --host 0.0.0.0 --port 8000 \
"]