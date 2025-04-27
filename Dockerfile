FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download and set up TwitchDownloaderCLI
RUN mkdir -p /app/bin
RUN wget -q https://github.com/lay295/TwitchDownloader/releases/download/1.55.0/TwitchDownloaderCLI-Linux-x64.zip -O /tmp/twitch-dl.zip \
    && unzip /tmp/twitch-dl.zip -d /app/bin \
    && chmod +x /app/bin/TwitchDownloaderCLI \
    && rm /tmp/twitch-dl.zip

# Create symbolic links
RUN ln -sf /app/bin/TwitchDownloaderCLI /app/TwitchDownloaderCLI \
    && ln -sf /usr/bin/ffmpeg /app/ffmpeg

# Set environment variables
ENV PATH="/app/bin:${PATH}"

# Cloud Functions uses the gunicorn webserver
ENV PORT=8080

# Use functions-framework to start the function
CMD exec functions-framework --target=run_pipeline --port=${PORT}
