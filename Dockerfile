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

# Make sure the binary files are executable
RUN chmod +x /app/raw_pipeline/bin/TwitchDownloaderCLI && \
    chmod +x /app/raw_pipeline/bin/ffmpeg

# Set environment variables
ENV PATH="/app/raw_pipeline/bin:${PATH}"
ENV DOTNET_BUNDLE_EXTRACT_BASE_DIR="/tmp/.dotnet/bundle_extract"
ENV PORT=8080

# Create necessary directories
RUN mkdir -p /tmp/output /tmp/downloads /tmp/data /tmp/logs

# Use functions-framework to start the function
CMD exec functions-framework --target=run_pipeline --port=${PORT}
