import logging
import time
from pathlib import Path
from typing import Any, Optional

from app.services.alignment_service import AlignmentService
from app.services.audio_service import AudioService
from app.services.cleanup_service import cleanup_file
from app.services.diarization_service import DiarizationService
from app.services.evaluation_service import EvaluationService
from app.services.summary_service import SummaryService
from app.services.whisper_service import WhisperService

logger = logging.getLogger("voice_notulen_api.meeting_service")


class MeetingService:
    """
    Orchestrator service for the full meeting transcription pipeline.
    Coordinates all sub-services and aggregates results.
    """

    def __init__(
        self,
        audio_service: AudioService,
        whisper_service: WhisperService,
        diarization_service: DiarizationService,
        alignment_service: AlignmentService,
        evaluation_service: EvaluationService,
        summary_service: SummaryService,
    ) -> None:
        self.audio = audio_service
        self.whisper = whisper_service
        self.diarization = diarization_service
        self.alignment = alignment_service
        self.evaluation = evaluation_service
        self.summary = summary_service

    def process(
        self,
        input_path: str | Path,
        reference_text: Optional[str] = None,
        uploaded_path: Optional[str | Path] = None,
    ) -> dict:
        """
        Execute the full meeting transcription pipeline.

        Pipeline:
            1. Audio preprocessing (FFmpeg → 16kHz WAV → normalize)
            2. Whisper transcription
            3. Pyannote diarization
            4. Alignment (chunks + speaker segments)
            5. Build combined speaker-aware transcript
            6. Gemini summary
            7. WER/CER evaluation
            8. Aggregate result and timings

        Args:
            input_path: Path to raw upload audio file.
            reference_text: Optional ground truth for WER/CER evaluation.
            uploaded_path: Original upload path for cleanup after processing.

        Returns:
            dict: {"transcript", "summary", "metrics", "timings"}
        """
        total_start = time.perf_counter()
        timings: dict[str, float] = {}
        processed_audio: Optional[str] = None

        try:
            # --- Stage 1: Audio Preprocessing ---
            t = time.perf_counter()
            processed_audio = self.audio.preprocess(input_path)
            timings["preprocessing"] = round(time.perf_counter() - t, 3)

            # --- Stage 2: Whisper Transcription ---
            t = time.perf_counter()
            whisper_result = self.whisper.transcribe(processed_audio)
            timings["whisper"] = round(time.perf_counter() - t, 3)

            # --- Stage 3: Pyannote Diarization ---
            t = time.perf_counter()
            diarization_segments = self.diarization.diarize(processed_audio)
            timings["diarization"] = round(time.perf_counter() - t, 3)

            # --- Stage 4: Alignment ---
            transcript_rows = self.alignment.align(
                whisper_result["chunks"],
                diarization_segments,
            )

            # --- Stage 5: Build combined speaker-aware transcript ---
            combined_text = "\n".join(
                f'{row["speaker"]}: {row["text"]}'
                for row in transcript_rows
                if row.get("text")
            )

            # --- Stage 6: Gemini Summary ---
            t = time.perf_counter()
            summary = self.summary.summarize(combined_text)
            timings["summary"] = round(time.perf_counter() - t, 3)

            # --- Stage 7: Evaluation ---
            metrics = self.evaluation.evaluate(
                reference_text,
                whisper_result["text"],
            )

            timings["total"] = round(time.perf_counter() - total_start, 3)

            logger.info(
                "[MEETING] pipeline completed timings=%s", timings
            )

            return {
                "status": "completed",
                "transcript": transcript_rows,
                "summary": summary,
                "metrics": {
                    **metrics,
                    "processing_time": timings["total"],
                },
                "timings": timings,
            }

        finally:
            # Cleanup processed WAV
            if processed_audio:
                cleanup_file(processed_audio)
            # Cleanup uploaded file if provided
            if uploaded_path:
                cleanup_file(uploaded_path)
