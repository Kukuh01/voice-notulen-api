"""
run_colab.py — Entry point for running Voice Notulen API on Google Colab.

Usage in Colab notebook cell:
    !git clone https://github.com/<username>/voice-notulen-api.git
    %cd voice-notulen-api
    !bash scripts/setup_colab.sh
    import run_colab

Or run directly:
    !python run_colab.py

The script:
1. Reads API keys from Colab userdata (secrets).
2. Writes them to environment variables.
3. Starts the FastAPI server with uvicorn.
4. Optionally exposes via ngrok or cloudflared tunnel.
"""
import os
import subprocess
import sys


def load_colab_secrets():
    """Load Gemini API key and HF token from Google Colab userdata."""
    try:
        from google.colab import userdata  # type: ignore

        gemini_key = userdata.get("GEMINI_API_KEY")
        hf_token = userdata.get("HF_TOKEN")

        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            print("[OK] GEMINI_API_KEY loaded from Colab secrets.")
        else:
            print("[WARN] GEMINI_API_KEY not found in Colab secrets.")

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            print("[OK] HF_TOKEN loaded from Colab secrets.")
        else:
            print("[WARN] HF_TOKEN not found in Colab secrets.")

    except ImportError:
        print("[INFO] Not running in Google Colab — using .env or environment variables.")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start FastAPI server with uvicorn."""
    print(f"\nStarting Voice Notulen API on http://{host}:{port}")
    print("Access API docs at: /docs")
    print("Press Ctrl+C to stop.\n")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    load_colab_secrets()
    start_server()
