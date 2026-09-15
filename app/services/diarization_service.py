import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("voice_notulen_api.diarization_service")


def _extract_segments(output: Any) -> list[dict]:
    """
    Adapter that normalizes Pyannote output across different API versions.
    Tries multiple attribute access strategies before failing.
    """
    # Strategy 1: result has .segments (e.g. dict-like objects)
    if hasattr(output, "segments"):
        segments = []
        for seg in output.segments:
            segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": str(seg.get("speaker", "SPEAKER_UNKNOWN")),
            })
        return segments

    # Strategy 2-4: look for named Annotation attributes
    annotation = None
    for attr_name in ("speaker_diarization", "diarization", "annotation"):
        candidate = getattr(output, attr_name, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            annotation = candidate
            break

    # Strategy 5: output itself is an Annotation
    if annotation is None and hasattr(output, "itertracks"):
        annotation = output

    if annotation is None:
        raise RuntimeError(
            "Unknown Pyannote output structure — cannot extract speaker segments. "
            "Attributes available: " + str(dir(output))
        )

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    return segments


class DiarizationService:
    """
    Service responsible for speaker diarization using Pyannote.
    Normalizes output to internal segment format for downstream services.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def diarize(self, audio_path: str | Path) -> list[dict]:
        """
        Run speaker diarization on processed WAV file.

        Returns:
            list[dict]: List of {"start", "end", "speaker"} segments.
        """
        audio_path = str(audio_path)
        logger.info("[DIARIZATION] started — %s", audio_path)
        t_start = time.perf_counter()

        output = self.model(audio_path)
        segments = _extract_segments(output)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "[DIARIZATION] finished duration=%.2fs segments=%d",
            elapsed,
            len(segments),
        )
        return segments
