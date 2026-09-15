from fastapi import Request

from app.services.meeting_service import MeetingService


def get_meeting_service(request: Request) -> MeetingService:
    """
    FastAPI dependency that retrieves the MeetingService singleton
    from application state. Raises 503 if models are not loaded.
    """
    meeting_service: MeetingService | None = getattr(
        request.app.state, "meeting_service", None
    )
    if meeting_service is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Meeting service is not available. Models may still be loading.",
        )
    return meeting_service
