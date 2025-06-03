FROM python:3.10-slim

# Install system dependencies including .NET runtime for TwitchDownloaderCLI
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    unzip \
    curl \
    gnupg \
    git \
    git-lfs \
    ca-certificates \
    libc6 \
    libgcc-s1 \
    libgssapi-krb5-2 \
    libicu72 \
    libssl3 \
    libstdc++6 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Install .NET 6.0 runtime (required for TwitchDownloaderCLI)
RUN wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y dotnet-runtime-6.0 && \
    rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Initialize Git LFS
RUN git lfs install

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install yt-dlp as a fallback for video downloads
RUN pip install --no-cache-dir yt-dlp

# Copy .gitattributes first to ensure LFS files are handled correctly
COPY .gitattributes .

# Copy application code (this will include LFS files)
COPY . .

# Ensure binary files are executable and create necessary directories
RUN mkdir -p /app/bin /tmp/.dotnet/bundle_extract && \
    # Make our LFS binary files executable (Linux versions for Cloud Run)
    chmod +x /app/raw_pipeline/bin/TwitchDownloaderCLI 2>/dev/null || echo "TwitchDownloaderCLI not found" && \
    chmod +x /app/raw_pipeline/bin/ffmpeg 2>/dev/null || echo "ffmpeg not found" && \
    # Verify the binary exists and is executable
    ls -la /app/raw_pipeline/bin/TwitchDownloaderCLI || echo "TwitchDownloaderCLI binary missing" && \
    file /app/raw_pipeline/bin/TwitchDownloaderCLI || echo "Cannot check TwitchDownloaderCLI file type" && \
    # Create symlink to system ffmpeg as backup
    ln -sf $(which ffmpeg) /app/raw_pipeline/bin/ffmpeg_system && \
    # Copy system ffmpeg as additional backup
    cp $(which ffmpeg) /app/bin/ffmpeg_system && \
    chmod +x /app/bin/ffmpeg_system

# Set environment variables - prioritize system binaries over bundled ones
ENV PATH="/usr/bin:/app/raw_pipeline/bin:${PATH}"
ENV DOTNET_BUNDLE_EXTRACT_BASE_DIR="/tmp/.dotnet/bundle_extract"
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Production optimizations
ENV UVICORN_WORKERS=1
ENV UVICORN_LOG_LEVEL=info
ENV UVICORN_ACCESS_LOG=true
ENV UVICORN_TIMEOUT_KEEP_ALIVE=65

# Create necessary directories
RUN mkdir -p /tmp/output /tmp/downloads /tmp/data /tmp/logs /tmp/output/Analysis/Chat /tmp/output/Analysis/Audio /tmp/output/Analysis/Integrated /tmp/output/Analysis/Integrated/editor_view

# Verify model files exist
RUN if [ ! -s /app/analysis_pipeline/chat/models/emotion_classifier_pipe_lr.pkl ] || [ ! -s /app/analysis_pipeline/chat/models/highlight_classifier_pipe_lr.pkl ]; then \
        echo "Warning: Model files are missing or empty. The application may not function correctly."; \
    else \
        echo "Model files verified successfully."; \
    fi

# Test TwitchDownloaderCLI binary
RUN echo "Testing TwitchDownloaderCLI binary..." && \
    /app/raw_pipeline/bin/TwitchDownloaderCLI --help > /dev/null 2>&1 && \
    echo "TwitchDownloaderCLI binary is working" || \
    echo "Warning: TwitchDownloaderCLI binary test failed"

# Add additional error handling for file operations
RUN echo "Adding retry logic for file operations"

# Health check endpoint for Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use uvicorn to start the FastAPI application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
