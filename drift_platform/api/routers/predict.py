from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def predict(payload: dict) -> dict:
    raise NotImplementedError
