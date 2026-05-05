"""Incoming webhook endpoints — triggers agent graph runs."""
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()


class DriftEvent(BaseModel):
    model_id: str
    psi_score: float
    triggered_at: str


@router.post("/drift")
async def on_drift_event(event: DriftEvent, background_tasks: BackgroundTasks) -> dict:
    from agent.graph.supervisor import run_graph
    background_tasks.add_task(run_graph, event.model_dump())
    return {"accepted": True}
