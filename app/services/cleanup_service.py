import logging
from pathlib import Path

logger = logging.getLogger("voice_notulen_api.cleanup_service")


def cleanup_file(path: str | Path) -> None:
    """
    Safely delete a temporary file. Logs error but does not raise
    exceptions to prevent cleanup failures from masking pipeline errors.
    """
    try:
        file_path = Path(path)
        if file_path.exists():
            file_path.unlink()
            logger.debug("Cleaned up temporary file: %s", file_path)
    except Exception as exc:
        logger.error("Failed to cleanup file %s: %s", path, exc)
