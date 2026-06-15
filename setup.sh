#!/usr/bin/env bash
set -e

echo "=== Chinese-to-Khmer Dubbing — Setup ==="

# 1. Clone Wav2Lip if not present
if [ ! -d "Wav2Lip" ]; then
  echo "[1/4] Cloning Wav2Lip..."
  git clone https://github.com/Rudrabha/Wav2Lip.git
else
  echo "[1/4] Wav2Lip already present, skipping clone."
fi

# 2. Download Wav2Lip GAN checkpoint
mkdir -p Wav2Lip/checkpoints
CKPT="Wav2Lip/checkpoints/wav2lip_gan.pth"
if [ ! -f "$CKPT" ]; then
  echo "[2/4] Downloading Wav2Lip GAN checkpoint..."
  echo "      Please download wav2lip_gan.pth from:"
  echo "      https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
  echo "      and place it at: $CKPT"
  echo "      (Direct download requires a Google Drive link — see the repo README)"
else
  echo "[2/4] Wav2Lip checkpoint already present."
fi

# 3. Copy .env
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "[3/4] Created .env — please add your ANTHROPIC_API_KEY."
else
  echo "[3/4] .env already exists."
fi

# 4. Install Python dependencies (local dev)
echo "[4/4] Installing Python backend dependencies..."
pip install -r backend/requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run locally (without Docker):"
echo "  1. Start Redis:  redis-server"
echo "  2. Start worker: cd backend && celery -A worker worker --loglevel=info"
echo "  3. Start API:    cd backend && uvicorn main:api --reload"
echo "  4. Open:         http://localhost:8000"
echo ""
echo "To run with Docker:"
echo "  docker compose up --build"
