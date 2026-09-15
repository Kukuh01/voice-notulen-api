#!/bin/bash
# setup_colab.sh — Setup Voice Notulen API on Google Colab GPU
# Usage: bash scripts/setup_colab.sh

set -e

echo "====================================================="
echo " Voice Notulen API — Colab Setup Script"
echo "====================================================="

# Install FFmpeg
echo "[1/4] Installing FFmpeg..."
apt-get install -y -q ffmpeg

# Install Python dependencies (excluding torch/torchaudio — Colab has them pre-installed)
echo "[2/4] Installing Python dependencies..."
pip install -q -r requirements-colab.txt

# Verify environment
echo "[3/4] Verifying environment..."
python scripts/check_environment.py

# Create storage directories
echo "[4/4] Creating storage directories..."
mkdir -p storage/uploads storage/processed storage/results

echo ""
echo "====================================================="
echo " Setup complete!"
echo " Run: python run_colab.py"
echo "====================================================="
