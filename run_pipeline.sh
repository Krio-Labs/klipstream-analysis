#!/bin/bash
# Script to clean up directories and run the main pipeline

# Check if a Twitch URL was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <twitch_video_url>"
    echo "Example: $0 https://www.twitch.tv/videos/2434635255"
    exit 1
fi

TWITCH_URL="$1"

echo "Starting cleanup process..."
python cleanup.py

echo "Running main pipeline with URL: $TWITCH_URL"
python main.py "$TWITCH_URL"

echo "Pipeline execution completed."
