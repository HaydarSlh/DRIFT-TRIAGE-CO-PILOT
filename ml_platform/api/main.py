from fastapi import FastAPI
from ml_platform.api.routers import predict, registry,retrain, replay

app = FastAPI(title="ML Platform")

app.include_router(predict.router)
app.include_router(registry.router)
app.include_router(retrain.router)
app.include_router(replay.router)

@app.get("/health")
def health():
    return {"status": "ok"}