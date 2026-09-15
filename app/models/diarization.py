from typing import Any, Optional
from app.core.config import Settings, get_settings


def load_diarization(settings: Optional[Settings] = None) -> Any:
    """
    Load Pyannote Speaker Diarization pipeline.
    Safely handles HuggingFace token and CUDA device placement.
    """
    import torch
    from pyannote.audio import Pipeline

    if settings is None:
        settings = get_settings()

    if not settings.hf_token:
        raise ValueError(
            "Hugging Face token (HF_TOKEN) is required to load Pyannote diarization model."
        )

    pipeline_obj = Pipeline.from_pretrained(
        settings.pyannote_model,
        token=settings.hf_token,
    )

    use_cuda = (
        torch.cuda.is_available() and settings.whisper_device.lower() == "cuda"
    )
    if use_cuda:
        pipeline_obj.to(torch.device("cuda"))

    return pipeline_obj
