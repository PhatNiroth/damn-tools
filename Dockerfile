FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support (separate for Docker layer caching)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install openai-whisper separately so setuptools is available in its build env
RUN pip install --no-cache-dir openai-whisper

# Pre-download Whisper models so they never hang at runtime
# (small is the default; tiny kept as a fast fallback via WHISPER_MODEL=tiny)
RUN python -c "import whisper; whisper.load_model('tiny'); whisper.load_model('small')"

# Khmer fonts for burned-in subtitles. Placed below the heavy torch/whisper
# layers so they stay cached when this changes.
#  - fonts-khmeros  : the "Khmer OS" font family the burn style requests
#                     (without it, libass renders Khmer as tofu boxes)
#  - fonts-noto-core: Noto Sans Khmer, a robust fallback with full shaping
#  - fontconfig     : lets ffmpeg/libass resolve the font by family name
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
    fonts-khmeros \
    fonts-noto-core \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# Install remaining dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: in-process VoxCPM2 for real voice cloning on a GPU box.
# Build with `--build-arg INSTALL_VOXCPM=1` on the GPU PC (skips on the laptop).
# torch is already the CUDA build installed above, so only `voxcpm` is added.
ARG INSTALL_VOXCPM=0
COPY backend/requirements-voxcpm.txt .
RUN if [ "$INSTALL_VOXCPM" = "1" ]; then \
        pip install --no-cache-dir voxcpm ; \
    fi

# Copy application code (bust cache with --no-cache or increment this comment: v2)
COPY backend/ /app/
COPY frontend/ /app/frontend/

RUN mkdir -p /app/uploads /app/outputs /app/data

EXPOSE 8000
