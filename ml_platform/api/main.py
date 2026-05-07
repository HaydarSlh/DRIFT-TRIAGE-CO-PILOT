from fastapi import FastAPI
from ml_platform.api.routers import predict, registry

app = FastAPI(title="ML Platform")

app.include_router(predict.router)
app.include_router(registry.router)

@app.get("/health")
def health():
    return {"status": "ok"}