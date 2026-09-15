import logging
import subprocess
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from app.core.config import Settings, get_settings
from app.utils.filesystem import (
    ensure_directory,
    generate_unique_filename,
    validate_file_not_empty,
)

logger = logging.getLogger("voice_notulen_api.audio_service")


class AudioService:
    """
    Service responsible for audio preprocessing:
    FFmpeg conversion -> 16 kHz mono PCM WAV -> amplitude normalization -> save.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or get_settings()

    def preprocess(self, input_path: str | Path) -> str:
        """
        Preprocess input audio file.

        Steps:
        1. Validate input file exists & not empty.
        2. Convert to WAV 16 kHz mono via FFmpeg.
        3. Validate WAV audio info (16000 Hz, 1 channel).
        4. Load audio, normalize amplitude.
        5. Save processed WAV to storage/processed directory.

        Returns:
            str: Path to processed WAV file.
        """
        input_file_path = validate_file_not_empty(input_path)
        logger.info("Starting audio preprocessing for: %s", input_file_path)

        processed_dir = ensure_directory(self.settings.processed_dir)
        output_filename = generate_unique_filename(extension="wav")
        output_path = processed_dir / output_filename

        # FFmpeg command: convert to 16kHz mono PCM 16-bit WAV
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        logger.info("Executing FFmpeg command for %s", input_file_path.name)
        res = subprocess.run(
            ffmpeg_cmd, capture_output=True, text=True, check=False
        )

        if res.returncode != 0:
            logger.error("FFmpeg conversion failed: %s", res.stderr)
            raise RuntimeError(
                f"FFmpeg conversion failed for file {input_file_path.name}: {res.stderr}"
            )

        # Validate WAV info
        info = sf.info(str(output_path))
        if info.samplerate != 16000 or info.channels != 1:
            logger.error(
                "FFmpeg output invalid: sr=%d, channels=%d",
                info.samplerate,
                info.channels,
            )
            raise ValueError(
                f"Converted WAV is invalid: samplerate={info.samplerate}, channels={info.channels}"
            )

        # Load audio using librosa for normalization
        audio, _ = librosa.load(str(output_path), sr=16000, mono=True)

        # Amplitude normalization
        max_amplitude = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if max_amplitude > 0:
            audio = audio / max_amplitude

        # Save processed normalized WAV
        sf.write(
            str(output_path),
            audio,
            16000,
            subtype="PCM_16",
        )

        logger.info("Audio preprocessing completed successfully: %s", output_path)
        return str(output_path)
