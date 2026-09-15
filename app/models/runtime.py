import logging
from typing import Any, Optional
from app.core.config import Settings, get_settings
from app.models.whisper import load_whisper
from app.models.diarization import load_diarization
from app.models.gemini import load_gemini

logger = logging.getLogger("voice_notulen_api.models")


class ModelRuntime:
    """
    Runtime container for AI models lifecycle management.
    Models are NOT loaded on initialization unless explicitly requested.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.whisper: Any = None
        self.diarization: Any = None
        self.gemini: Any = None

    def initialize(
        self,
        load_whisper_model: bool = True,
        load_diarization_model: bool = True,
        load_gemini_client: bool = True,
    ) -> None:
        """
        Explicitly load AI models into runtime memory.
        """
        if load_whisper_model and self.whisper is None:
            logger.info("Loading Whisper model: %s", self.settings.whisper_model)
            self.whisper = load_whisper(self.settings)
            logger.info("Whisper model loaded successfully.")

        if load_diarization_model and self.diarization is None:
            logger.info("Loading Pyannote model: %s", self.settings.pyannote_model)
            self.diarization = load_diarization(self.settings)
            logger.info("Pyannote model loaded successfully.")

        if load_gemini_client and self.gemini is None:
            logger.info("Initializing Gemini client.")
            self.gemini = load_gemini(self.settings)
            logger.info("Gemini client initialized successfully.")


_runtime_instance: Optional[ModelRuntime] = None


def get_model_runtime(settings: Optional[Settings] = None) -> ModelRuntime:
    """
    Get singleton instance of ModelRuntime container.
    Optionally accepts Settings to pass on first creation.
    """
    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = ModelRuntime(settings)
    return _runtime_instance
