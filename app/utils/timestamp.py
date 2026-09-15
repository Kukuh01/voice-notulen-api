def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to H:MM:SS string format.

    Examples:
        format_timestamp(0)    → "0:00:00"
        format_timestamp(61)   → "0:01:01"
        format_timestamp(3661) → "1:01:01"
    """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"
