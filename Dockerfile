FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    unzip \
    curl \
    gnupg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs && \
    git lfs install

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install ffmpeg and create a simple wrapper script for TwitchDownloaderCLI
RUN mkdir -p /app/raw_pipeline/bin && \
    apt-get update && apt-get install -y ffmpeg && \
    cp $(which ffmpeg) /app/raw_pipeline/bin/ffmpeg && \
    chmod +x /app/raw_pipeline/bin/ffmpeg && \
    echo '#!/bin/bash\necho "TwitchDownloaderCLI mock: $@"\nexit 0' > /app/raw_pipeline/bin/TwitchDownloaderCLI && \
    chmod +x /app/raw_pipeline/bin/TwitchDownloaderCLI

# Make sure the binary files are executable
RUN chmod +x /app/raw_pipeline/bin/TwitchDownloaderCLI || true && \
    chmod +x /app/raw_pipeline/bin/ffmpeg || true

# Set environment variables
ENV PATH="/app/raw_pipeline/bin:${PATH}"
ENV DOTNET_BUNDLE_EXTRACT_BASE_DIR="/tmp/.dotnet/bundle_extract"
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Create necessary directories
RUN mkdir -p /tmp/output /tmp/downloads /tmp/data /tmp/logs

# Verify model files exist
RUN if [ ! -s /app/analysis_pipeline/chat/models/emotion_classifier_pipe_lr.pkl ] || [ ! -s /app/analysis_pipeline/chat/models/highlight_classifier_pipe_lr.pkl ]; then \
        echo "Warning: Model files are missing or empty. The application may not function correctly."; \
    else \
        echo "Model files verified successfully."; \
    fi

# Use functions-framework to start the function
CMD ["functions-framework", "--target=run_pipeline", "--port=8080"]
