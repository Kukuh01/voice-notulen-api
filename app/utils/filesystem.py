import os
import uuid
from pathlib import Path


def ensure_directory(dir_path: str | Path) -> Path:
    """Ensure directory exists on the filesystem."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(file_path: str | Path) -> Path:
    """Validate file exists on filesystem."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path


def validate_file_not_empty(file_path: str | Path) -> Path:
    """Validate file exists and is not empty (0 bytes)."""
    path = validate_file_exists(file_path)
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty (0 bytes): {file_path}")
    return path


def generate_unique_filename(prefix: str = "", extension: str = "wav") -> str:
    """Generate a unique filename using UUID4."""
    ext = extension.lstrip(".")
    unique_id = uuid.uuid4().hex
    if prefix:
        return f"{prefix}_{unique_id}.{ext}"
    return f"{unique_id}.{ext}"
