#!/bin/bash
# Script to run the raw pipeline for downloading and uploading raw files

# Check if a Twitch URL was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <twitch_video_url>"
    echo "Example: $0 https://www.twitch.tv/videos/2434635255"
    exit 1
fi

TWITCH_URL="$1"

echo "Starting raw pipeline for URL: $TWITCH_URL"
python raw_pipeline.py "$TWITCH_URL"

echo "Raw pipeline execution completed."
