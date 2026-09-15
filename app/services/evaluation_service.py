import logging

from jiwer import cer, wer

logger = logging.getLogger("voice_notulen_api.evaluation_service")


class EvaluationService:
    """
    Service responsible for computing WER (Word Error Rate) and
    CER (Character Error Rate) when reference text is available.
    """

    def evaluate(
        self,
        reference_text: str | None,
        prediction_text: str,
    ) -> dict:
        """
        Compute WER and CER against reference text.

        Args:
            reference_text: Ground truth transcription. If None or empty,
                            evaluation is skipped and both scores are None.
            prediction_text: Full Whisper prediction text (not speaker-aligned).

        Returns:
            dict: {"wer": float | None, "cer": float | None}
        """
        if not reference_text or not reference_text.strip():
            logger.info("[EVALUATION] skipped — no reference text provided")
            return {"wer": None, "cer": None}

        wer_score = float(wer(reference_text, prediction_text))
        cer_score = float(cer(reference_text, prediction_text))

        logger.info(
            "[EVALUATION] WER=%.4f CER=%.4f", wer_score, cer_score
        )
        return {"wer": wer_score, "cer": cer_score}
