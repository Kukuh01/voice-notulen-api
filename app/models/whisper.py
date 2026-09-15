from typing import Any, Optional
from app.core.config import Settings, get_settings


def load_whisper(settings: Optional[Settings] = None) -> Any:
    """
    Load Hugging Face Whisper automatic-speech-recognition pipeline.
    Safely detects CUDA vs CPU availability and sets appropriate torch_dtype.
    """
    import torch
    from transformers import pipeline

    if settings is None:
        settings = get_settings()

    use_cuda = (
        torch.cuda.is_available() and settings.whisper_device.lower() == "cuda"
    )
    device = 0 if use_cuda else -1

    # CPU backend does not support float16 for many torch operations; fallback to float32 on CPU
    if use_cuda and settings.whisper_dtype.lower() == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=settings.whisper_model,
        device=device,
        torch_dtype=torch_dtype,
    )
    return whisper_pipe
