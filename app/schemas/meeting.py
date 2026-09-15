from typing import Optional

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """A single aligned transcript chunk with speaker label."""
    start: float
    end: float
    speaker: str
    text: str


class StageTiming(BaseModel):
    """Per-stage processing duration in seconds."""
    preprocessing: Optional[float] = None
    whisper: Optional[float] = None
    diarization: Optional[float] = None
    summary: Optional[float] = None
    total: Optional[float] = None


class ProcessingMetrics(BaseModel):
    """WER/CER scores and processing time."""
    wer: Optional[float] = None
    cer: Optional[float] = None
    processing_time: float


class MeetingResult(BaseModel):
    """Full API response for a meeting transcription request."""
    status: str
    transcript: list[TranscriptSegment]
    summary: str
    metrics: ProcessingMetrics
    timings: Optional[StageTiming] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str


class JobResponse(BaseModel):
    """
    Placeholder for future async job response.
    Not used in synchronous MVP.
    """
    job_id: str
    status: str
