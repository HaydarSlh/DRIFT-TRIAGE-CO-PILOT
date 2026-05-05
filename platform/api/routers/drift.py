from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_drift_reports() -> list:
    raise NotImplementedError
