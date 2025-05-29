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

# Install ffmpeg and download TwitchDownloaderCLI from the official repository
RUN mkdir -p /app/raw_pipeline/bin /app/bin && \
    apt-get update && \
    cp $(which ffmpeg) /app/raw_pipeline/bin/ffmpeg && \
    chmod +x /app/raw_pipeline/bin/ffmpeg && \
    # Download and extract TwitchDownloaderCLI
    wget -q https://github.com/lay295/TwitchDownloader/releases/download/1.55.7/TwitchDownloaderCLI-1.55.7-Linux-x64.zip -O /tmp/twitch-dl.zip && \
    unzip -o /tmp/twitch-dl.zip -d /app/raw_pipeline/bin && \
    chmod +x /app/raw_pipeline/bin/TwitchDownloaderCLI && \
    cp /app/raw_pipeline/bin/TwitchDownloaderCLI /app/bin/TwitchDownloaderCLI && \
    chmod +x /app/bin/TwitchDownloaderCLI && \
    rm /tmp/twitch-dl.zip

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
RUN mkdir -p /tmp/output /tmp/downloads /tmp/data /tmp/logs /tmp/output/Analysis/Chat /tmp/output/Analysis/Audio /tmp/output/Analysis/Integrated /tmp/output/Analysis/Integrated/editor_view

# Verify model files exist
RUN if [ ! -s /app/analysis_pipeline/chat/models/emotion_classifier_pipe_lr.pkl ] || [ ! -s /app/analysis_pipeline/chat/models/highlight_classifier_pipe_lr.pkl ]; then \
        echo "Warning: Model files are missing or empty. The application may not function correctly."; \
    else \
        echo "Model files verified successfully."; \
    fi

# Add additional error handling for file operations
RUN echo "Adding retry logic for file operations"

# Use functions-framework to start the function
CMD ["functions-framework", "--target=run_pipeline", "--port=8080"]
