from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_models() -> list:
    raise NotImplementedError
