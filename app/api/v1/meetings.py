import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_meeting_service
from app.core.config import get_settings
from app.schemas.meeting import MeetingResult
from app.services.meeting_service import MeetingService
from app.utils.filesystem import ensure_directory, generate_unique_filename

logger = logging.getLogger("voice_notulen_api.api.meetings")

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".mp4"}

router = APIRouter(prefix="/meetings", tags=["Meetings"])


@router.post(
    "/transcribe",
    response_model=MeetingResult,
    summary="Transcribe meeting audio",
    description=(
        "Upload an audio file of a meeting recording. "
        "The API will transcribe, diarize, align speakers, optionally evaluate "
        "against reference text, and generate a structured Gemini meeting summary."
    ),
)
async def transcribe_meeting(
    file: UploadFile = File(..., description="Audio file (MP3, WAV, M4A, etc.)"),
    reference_text: Optional[str] = Form(
        None,
        description="Optional reference transcript for WER/CER evaluation.",
    ),
    meeting_service: MeetingService = Depends(get_meeting_service),
) -> MeetingResult:
    """
    POST /api/v1/meetings/transcribe

    Accepts multipart/form-data:
    - file: audio file
    - reference_text: optional string for WER/CER evaluation

    Returns MeetingResult with transcript, summary, and metrics.
    """
    settings = get_settings()

    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File size {size_mb:.1f} MB exceeds limit of {settings.max_upload_size_mb} MB.",
        )

    # Save to storage/uploads/<uuid>.<ext>
    upload_dir = ensure_directory(settings.upload_dir)
    upload_filename = generate_unique_filename(extension=suffix.lstrip("."))
    upload_path = upload_dir / upload_filename

    try:
        upload_path.write_bytes(content)
        logger.info(
            "Saved upload: %s (%.1f MB)", upload_path.name, size_mb
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        ) from exc

    # Run full pipeline through orchestrator
    try:
        result = meeting_service.process(
            input_path=upload_path,
            reference_text=reference_text,
            uploaded_path=upload_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Internal pipeline error: {exc}",
        ) from exc

    return MeetingResult(**result)
