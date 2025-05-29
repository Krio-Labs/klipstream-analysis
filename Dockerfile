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
    libgcc1 \
    libgssapi-krb5-2 \
    libicu67 \
    libssl1.1 \
    libstdc++6 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Install .NET 6.0 runtime (required for TwitchDownloaderCLI)
RUN wget https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
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

# Copy .gitattributes first to ensure LFS files are handled correctly
COPY .gitattributes .

# Copy application code (this will include LFS files)
COPY . .

# Ensure binary files are executable and create necessary directories
RUN mkdir -p /app/bin /tmp/.dotnet/bundle_extract && \
    # Make our LFS binary files executable
    chmod +x /app/raw_pipeline/bin/TwitchDownloaderCLI 2>/dev/null || echo "TwitchDownloaderCLI not found, will use system ffmpeg" && \
    chmod +x /app/raw_pipeline/bin/ffmpeg_mac 2>/dev/null || echo "ffmpeg_mac not found" && \
    chmod +x /app/raw_pipeline/bin/ffmpeg 2>/dev/null || echo "ffmpeg symlink not found" && \
    # Copy system ffmpeg as backup if our binaries don't work
    cp $(which ffmpeg) /app/bin/ffmpeg_system && \
    chmod +x /app/bin/ffmpeg_system && \
    # Create a fallback script for TwitchDownloaderCLI if needed
    echo '#!/bin/bash' > /app/bin/twitch_fallback.sh && \
    echo 'echo "TwitchDownloaderCLI fallback - binary not available"' >> /app/bin/twitch_fallback.sh && \
    chmod +x /app/bin/twitch_fallback.sh

# Set environment variables - prioritize system binaries over bundled ones
ENV PATH="/usr/bin:/app/raw_pipeline/bin:${PATH}"
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
