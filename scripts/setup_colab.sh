#!/bin/bash
set -e

echo "====================================================="
echo " Voice Notulen API — Colab Setup Script"
echo "====================================================="

echo "[1/4] Installing FFmpeg..."
apt-get update -qq && apt-get install -y -qq ffmpeg

echo "[2/4] Upgrading pyannote.audio & Installing Dependencies..."
pip install -q --upgrade "pyannote.audio>=3.3.1"
pip install -q -r requirements-colab.txt

echo "[3/4] Creating storage directories..."
mkdir -p storage/uploads storage/processed storage/results

echo "[4/4] Verifying environment..."
python scripts/check_environment.py

echo ""
echo "====================================================="
echo " Setup complete!"
echo "====================================================="