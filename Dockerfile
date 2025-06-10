# GPU-Optimized Dockerfile for KlipStream Analysis
# Supports NVIDIA L4 GPU for Parakeet transcription

FROM python:3.10-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including CUDA toolkit
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    unzip \
    software-properties-common \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install Cython first (required for NeMo dependencies)
RUN pip install Cython

# Install NeMo with ASR support
RUN pip install nemo_toolkit[asr]==1.20.0

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install additional GPU-specific dependencies
RUN pip install \
    nvidia-ml-py3 \
    pynvml \
    accelerate \
    transformers \
    datasets \
    librosa \
    soundfile \
    scipy \
    scikit-learn

# Copy TwitchDownloaderCLI binaries from raw_pipeline/bin
COPY raw_pipeline/bin/TwitchDownloaderCLI /app/raw_pipeline/bin/TwitchDownloaderCLI
COPY raw_pipeline/bin/TwitchDownloaderCLI_mac /app/raw_pipeline/bin/TwitchDownloaderCLI_mac
COPY raw_pipeline/bin/TwitchDownloaderCLI.exe /app/raw_pipeline/bin/TwitchDownloaderCLI.exe
COPY raw_pipeline/bin/ffmpeg /app/raw_pipeline/bin/ffmpeg
COPY raw_pipeline/bin/ffmpeg_mac /app/raw_pipeline/bin/ffmpeg_mac
COPY raw_pipeline/bin/ffmpeg.exe /app/raw_pipeline/bin/ffmpeg.exe

# Make binaries executable
RUN chmod +x /app/raw_pipeline/bin/TwitchDownloaderCLI
RUN chmod +x /app/raw_pipeline/bin/TwitchDownloaderCLI_mac
RUN chmod +x /app/raw_pipeline/bin/ffmpeg
RUN chmod +x /app/raw_pipeline/bin/ffmpeg_mac

# Copy application code
COPY . /app/

# Set GPU environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# GPU Transcription Configuration
ENV ENABLE_GPU_TRANSCRIPTION=true
ENV PARAKEET_MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2
ENV GPU_BATCH_SIZE=8
ENV GPU_MEMORY_LIMIT_GB=20
ENV TRANSCRIPTION_METHOD=auto
ENV ENABLE_FALLBACK=true
ENV COST_OPTIMIZATION=true

# Performance Configuration
ENV CHUNK_DURATION_MINUTES=10
ENV MAX_CONCURRENT_CHUNKS=4
ENV ENABLE_BATCH_PROCESSING=true
ENV ENABLE_PERFORMANCE_METRICS=true

# Cloud Run Configuration
ENV CLOUD_RUN_TIMEOUT_SECONDS=3600
ENV MAX_FILE_SIZE_GB=2

# Create necessary directories
RUN mkdir -p /app/output/raw /app/output/analysis /app/output/transcripts /app/output/cost_tracking
RUN mkdir -p /tmp/chunks /tmp/models

# Copy health check script
COPY health_check.py /app/health_check.py

RUN chmod +x /app/health_check.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 /app/health_check.py || exit 1

# Expose port
EXPOSE 8080

# Set user (Cloud Run requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Pre-download model to reduce cold start time (optional)
# RUN python3 -c "
# import os
# if os.getenv('ENABLE_GPU_TRANSCRIPTION', 'false').lower() == 'true':
#     try:
#         import nemo.collections.asr as nemo_asr
#         print('Pre-downloading Parakeet model...')
#         model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')
#         print('Model pre-downloaded successfully')
#     except Exception as e:
#         print(f'Model pre-download failed: {e}')
# "

# Start FastAPI server for Cloud Run
CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
