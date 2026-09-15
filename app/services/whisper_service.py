import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("voice_notulen_api.whisper_service")


class WhisperService:
    """
    Service responsible for audio transcription using Whisper Large-v3.
    Normalizes output to internal chunk format for downstream services.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def transcribe(self, audio_path: str | Path) -> dict:
        """
        Transcribe audio file using Whisper pipeline.

        Returns:
            dict:
                - "text" (str): Full concatenated transcription.
                - "chunks" (list[dict]): List of {"start", "end", "text"} dicts.
        """
        audio_path = str(audio_path)
        logger.info("[WHISPER] started — %s", audio_path)
        t_start = time.perf_counter()

        result = self.model(
            audio_path,
            chunk_length_s=15,
            batch_size=8,
            return_timestamps=True,
            generate_kwargs={
                "language": "id",
                "task": "transcribe",
                "condition_on_prev_tokens": True,
            },
        )

        elapsed = time.perf_counter() - t_start
        logger.info("[WHISPER] finished duration=%.2fs", elapsed)

        chunks = self._normalize_chunks(result.get("chunks", []))
        full_text: str = result.get("text", "").strip()

        return {
            "text": full_text,
            "chunks": chunks,
        }

    @staticmethod
    def _normalize_chunks(raw_chunks: list) -> list[dict]:
        """
        Normalize Whisper pipeline chunk output to internal format.
        Each chunk becomes: {"start": float, "end": float, "text": str}
        """
        normalized = []
        for chunk in raw_chunks:
            timestamp = chunk.get("timestamp", (None, None))
            start = float(timestamp[0]) if timestamp[0] is not None else 0.0
            end = float(timestamp[1]) if timestamp[1] is not None else start
            text = chunk.get("text", "").strip()
            normalized.append({"start": start, "end": end, "text": text})
        return normalized
