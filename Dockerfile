# Inference container for the Neyshekar Persian ASR model.
#
#   docker build -t neyshekar-asr .
#   docker run -p 8000:8000 neyshekar-asr
#   curl -F "file=@clip.wav" http://localhost:8000/transcribe
#
# Two things are deliberately NOT in this image.
#
# The base model. openai/whisper-large-v3 is about 3 GB; baking it in would make the image
# unwieldy to build, push and pull. It is fetched on first start and cached, so mounting a
# volume at /cache keeps it between runs (see the run command in the README).
#
# The dataset. Inference does not need it. Nothing under data/ or run_artifacts/ is copied,
# which .dockerignore enforces.
#
# The image is CPU-only. bitsandbytes 4-bit quantisation needs CUDA, and most machines that
# will run this do not have a GPU; serve.py detects that and loads float32 instead. For a GPU
# host, switch the base image to a CUDA runtime and add bitsandbytes to requirements-docker.txt
# — serve.py already takes the quantised path when torch.cuda.is_available() returns true.

FROM python:3.11-slim

# libsndfile is what soundfile binds to for decoding; ffmpeg lets librosa read formats
# soundfile cannot, such as mp3 and m4a. Neither has a pure-Python substitute.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so editing serve.py does not invalidate the layer that took
# several minutes to install.
COPY requirements-docker.txt .

# torch from PyPI is the CUDA build: it pulls roughly 2.5 GB of nvidia-* wheels that a CPU-only
# image can never use. Installing it from PyTorch's CPU index first cuts the finished image from
# about 5 GB to under 1.5 GB, and pip then treats the pinned requirement below as satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && find /usr/local/lib/python3.11 -name "__pycache__" -type d -prune -exec rm -rf {} +

COPY src/ ./src/
COPY serve.py ./

# HuggingFace writes to ~/.cache by default. Pointing it at an explicit directory means a
# single -v mount keeps the 3 GB base model across container restarts.
ENV HF_HOME=/cache \
    PYTHONUNBUFFERED=1 \
    ADAPTER_ID=hosseinzr/neyshekar-whisper-large-v3-lora \
    BASE_MODEL=openai/whisper-large-v3
RUN mkdir -p /cache

EXPOSE 8000

# The health check reflects whether the MODEL loaded, not merely whether the port is open —
# /health returns 503 until it has. Start period is long because the first run downloads 3 GB.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
