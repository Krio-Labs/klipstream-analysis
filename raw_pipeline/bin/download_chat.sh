#!/bin/bash

# This script downloads chat for a Twitch VOD without embedding images
# Usage: ./download_chat.sh <video_id> <output_path>

VIDEO_ID=$1
OUTPUT_PATH=$2

# Set environment variable for macOS
export DOTNET_BUNDLE_EXTRACT_BASE_DIR=/tmp

# Run the TwitchDownloaderCLI with the correct parameters
./raw_pipeline/bin/TwitchDownloaderCLI_mac chatdownload \
  --id "$VIDEO_ID" \
  --output "$OUTPUT_PATH" \
  --embed-images false \
  --bttv false \
  --ffz false \
  --stv false \
  --timestamp-format Relative \
  --log-level Status,Error \
  --threads 32 \
  --collision Overwrite
