"""Webhook receiver — entry point for the agent service."""
from fastapi import FastAPI
from agent.api.routes import webhooks

app = FastAPI(title="Drift Triage Agent")

app.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
