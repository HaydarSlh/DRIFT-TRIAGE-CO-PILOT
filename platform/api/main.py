from fastapi import FastAPI
from platform.api.routers import predict, drift, registry

app = FastAPI(title="Drift Triage Platform")

app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(drift.router, prefix="/drift", tags=["drift"])
app.include_router(registry.router, prefix="/registry", tags=["registry"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
