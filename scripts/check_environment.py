"""
Environment verification script for Voice Notulen API.
Run this before starting the API to validate all dependencies are correctly installed.

Usage:
    python scripts/check_environment.py
"""
import subprocess
import sys
import torchaudio


def check_python():
    version = sys.version_info
    ok = version >= (3, 11)
    tag = "OK" if ok else "WARN"
    print(f"[{tag}] Python {version.major}.{version.minor}.{version.micro}")
    return ok


def check_pytorch():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"[OK] PyTorch {torch.__version__} — CUDA available: {cuda_available}")
        if cuda_available:
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("     [WARN] No CUDA GPU detected. Models will run on CPU (slow).")
        return True
    except ImportError:
        print("[FAIL] PyTorch is not installed.")
        return False


def check_transformers():
    try:
        import transformers
        print(f"[OK] Transformers {transformers.__version__}")
        return True
    except ImportError:
        print("[FAIL] transformers is not installed.")
        return False


def check_pyannote():
    try:
        import pyannote.audio
        print(f"[OK] pyannote.audio {pyannote.audio.__version__}")
        return True
    except ImportError:
        print("[FAIL] pyannote.audio is not installed.")
        return False


def check_ffmpeg():
    result = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True
    )
    if result.returncode == 0:
        version_line = result.stdout.splitlines()[0]
        print(f"[OK] FFmpeg — {version_line}")
        return True
    else:
        print("[FAIL] FFmpeg is not available in PATH.")
        return False


def check_google_genai():
    try:
        import google.genai
        print(f"[OK] google-genai imported successfully")
        return True
    except ImportError:
        print("[FAIL] google-genai is not installed.")
        return False


def check_librosa():
    try:
        import librosa
        print(f"[OK] librosa {librosa.__version__}")
        return True
    except ImportError:
        print("[FAIL] librosa is not installed.")
        return False


def check_soundfile():
    try:
        import soundfile
        print(f"[OK] soundfile {soundfile.__version__}")
        return True
    except ImportError:
        print("[FAIL] soundfile is not installed.")
        return False


def check_jiwer():
    try:
        import jiwer
        print(f"[OK] jiwer imported successfully")
        return True
    except ImportError:
        print("[FAIL] jiwer is not installed.")
        return False


def check_app_config():
    try:
        from app.core.config import get_settings
        settings = get_settings()
        has_gemini = bool(settings.gemini_api_key)
        has_hf = bool(settings.hf_token)
        print(f"[OK] App config loaded — GEMINI_API_KEY={'SET' if has_gemini else 'NOT SET'}, HF_TOKEN={'SET' if has_hf else 'NOT SET'}")
        if not has_gemini:
            print("     [WARN] GEMINI_API_KEY not set — Gemini summary will fail at runtime.")
        if not has_hf:
            print("     [WARN] HF_TOKEN not set — Pyannote will fail to load.")
        return True
    except Exception as exc:
        print(f"[FAIL] App config error: {exc}")
        return False


def main():
    print("=" * 60)
    print("Voice Notulen API — Environment Check")
    print("=" * 60)

    checks = [
        check_python,
        check_pytorch,
        check_transformers,
        check_pyannote,
        check_ffmpeg,
        check_google_genai,
        check_librosa,
        check_soundfile,
        check_jiwer,
        check_app_config,
    ]

    results = [check() for check in checks]

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed.")
    if all(results):
        print("Environment is ready to run Voice Notulen API.")
    else:
        print("Some checks failed. Please resolve issues before starting the API.")
    print("=" * 60)


if __name__ == "__main__":
    main()

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None
