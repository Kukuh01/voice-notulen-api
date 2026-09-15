import logging

logger = logging.getLogger("voice_notulen_api.alignment_service")


class AlignmentService:
    """
    Service responsible for aligning Whisper transcript chunks with Pyannote
    speaker segments using maximum overlap algorithm from prototype.
    """

    def align(
        self,
        whisper_chunks: list[dict],
        diarization_segments: list[dict],
    ) -> list[dict]:
        """
        Assign speaker label to each Whisper chunk based on maximum overlap
        with diarization segments.

        Args:
            whisper_chunks: List of {"start", "end", "text"} dicts.
            diarization_segments: List of {"start", "end", "speaker"} dicts.

        Returns:
            list[dict]: List of {"start", "end", "speaker", "text"} dicts.
        """
        rows = []

        for chunk in whisper_chunks:
            chunk_start = float(chunk.get("start", 0.0))
            chunk_end = float(chunk.get("end", chunk_start))
            text = chunk.get("text", "").strip()

            best_speaker = "SPEAKER UNKNOWN"
            max_overlap = 0.0

            for segment in diarization_segments:
                seg_start = float(segment.get("start", 0.0))
                seg_end = float(segment.get("end", seg_start))

                overlap_start = max(chunk_start, seg_start)
                overlap_end = min(chunk_end, seg_end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = segment.get("speaker", "SPEAKER UNKNOWN")

            rows.append({
                "start": chunk_start,
                "end": chunk_end,
                "speaker": best_speaker,
                "text": text,
            })

        logger.info(
            "[ALIGNMENT] aligned %d chunks with %d diarization segments",
            len(whisper_chunks),
            len(diarization_segments),
        )
        return rows
