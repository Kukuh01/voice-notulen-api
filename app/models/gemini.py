from typing import Any, Optional
from app.core.config import Settings, get_settings


def load_gemini(settings: Optional[Settings] = None) -> Any:
    """
    Initialize Google Gemini Client using google-genai package.
    """
    from google import genai

    if settings is None:
        settings = get_settings()

    if not settings.gemini_api_key:
        raise ValueError(
            "Gemini API key (GEMINI_API_KEY) is required to load Gemini client."
        )

    return genai.Client(api_key=settings.gemini_api_key)
