from functools import lru_cache
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    Supports Google Colab userdata fallback for API keys/tokens.
    """

    app_name: str = "Voice Notulen API"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    gemini_api_key: str = ""
    hf_token: str = ""

    whisper_model: str = "openai/whisper-large-v3"
    whisper_language: str = "id"
    whisper_device: str = "cuda"
    whisper_dtype: str = "float16"

    pyannote_model: str = "pyannote/speaker-diarization-3.1"

    upload_dir: str = "storage/uploads"
    processed_dir: str = "storage/processed"
    result_dir: str = "storage/results"
    max_upload_size_mb: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _check_colab_secrets(self) -> "Settings":
        """
        Fallback check for Google Colab environment secrets
        if environment variables are not set.
        """
        if not self.gemini_api_key or not self.hf_token:
            try:
                from google.colab import userdata  # type: ignore

                if not self.gemini_api_key:
                    try:
                        val = userdata.get("GEMINI_API_KEY")
                        if val:
                            self.gemini_api_key = val
                    except Exception:
                        pass

                if not self.hf_token:
                    try:
                        val = userdata.get("HF_TOKEN")
                        if val:
                            self.hf_token = val
                    except Exception:
                        pass
            except ImportError:
                pass
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Get cached instance of Settings.
    """
    return Settings()
