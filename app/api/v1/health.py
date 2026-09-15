from fastapi import APIRouter

from app.schemas.meeting import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Returns 200 OK if the API is running."""
    return HealthResponse(status="ok")
