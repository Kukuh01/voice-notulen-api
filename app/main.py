import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.models.runtime import get_model_runtime
from app.services.alignment_service import AlignmentService
from app.services.audio_service import AudioService
from app.services.diarization_service import DiarizationService
from app.services.evaluation_service import EvaluationService
from app.services.meeting_service import MeetingService
from app.services.summary_service import SummaryService
from app.services.whisper_service import WhisperService
from app.utils.filesystem import ensure_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("voice_notulen_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle manager."""
    settings = get_settings()
    logger.info("Starting %s [env=%s]", settings.app_name, settings.app_env)

    # Ensure storage directories exist
    for dir_path in (settings.upload_dir, settings.processed_dir, settings.result_dir):
        ensure_directory(dir_path)
        logger.info("Storage directory ready: %s", dir_path)

    # Load AI models
    runtime = get_model_runtime(settings)
    runtime.initialize(
        load_whisper_model=True,
        load_diarization_model=True,
        load_gemini_client=True,
    )

    # Build and store MeetingService
    meeting_service = MeetingService(
        audio_service=AudioService(settings),
        whisper_service=WhisperService(runtime.whisper),
        diarization_service=DiarizationService(runtime.diarization),
        alignment_service=AlignmentService(),
        evaluation_service=EvaluationService(),
        summary_service=SummaryService(runtime.gemini),
    )
    app.state.meeting_service = meeting_service
    logger.info("All models loaded — API is ready.")

    yield

    # Shutdown
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Backend API for meeting voice recognition, speaker diarization, "
            "and AI-powered meeting summary using Whisper Large-v3, "
            "Pyannote 3.1, and Gemini 2.5 Flash."
        ),
        lifespan=lifespan,
    )
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_app()
