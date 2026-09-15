import logging
from typing import Any

logger = logging.getLogger("voice_notulen_api.summary_service")

SUMMARY_PROMPT_TEMPLATE = """Kamu adalah asisten notulensi rapat profesional.

Berdasarkan transkripsi rapat berikut, buat notulensi yang terstruktur dalam format:

RINGKASAN RAPAT
[Ringkasan singkat poin-poin utama yang dibahas dalam rapat]

KEPUTUSAN RAPAT
1. [Keputusan pertama]
2. [Keputusan kedua]
(Jika tidak ada keputusan eksplisit, tulis "Tidak ada keputusan formal yang diambil")

ACTION ITEMS
[Nama/Peran]:
- [Tugas yang harus dikerjakan]
- [Tenggat waktu jika disebutkan]
(Jika tidak ada action items, tulis "Tidak ada action items yang ditetapkan")

Transkripsi rapat:
{transcript}

Buat notulensi dalam Bahasa Indonesia yang formal dan ringkas."""


class SummaryService:
    """
    Service responsible for generating meeting summaries using Gemini API.
    """

    def __init__(self, client: Any, model_name: str = "gemini-2.5-flash") -> None:
        self.client = client
        self.model_name = model_name

    def summarize(self, transcript_text: str) -> str:
        """
        Generate structured meeting summary from speaker-aware transcript.

        Args:
            transcript_text: Transcript with speaker labels, e.g.
                             "SPEAKER_00: agenda rapat...\nSPEAKER_01: ..."

        Returns:
            str: Structured meeting summary in Indonesian.

        Raises:
            ValueError: If transcript_text is empty.
            RuntimeError: If Gemini API call fails.
        """
        if not transcript_text or not transcript_text.strip():
            raise ValueError("Transcript text is empty — cannot generate summary.")

        prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript_text.strip())

        logger.info("[GEMINI] started — generating meeting summary")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            summary = response.text
            logger.info("[GEMINI] finished — summary generated successfully")
            return summary
        except Exception as exc:
            logger.error("[GEMINI] failed — error=%s", exc)
            raise RuntimeError(f"Gagal membuat ringkasan: {exc}") from exc
