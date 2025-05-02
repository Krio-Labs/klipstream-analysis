#!/bin/bash
# Script to clean up directories and run the main pipeline

# Check if a Twitch URL was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <twitch_video_url>"
    echo "Example: $0 https://www.twitch.tv/videos/2434635255"
    exit 1
fi

TWITCH_URL="$1"

# Set up .NET bundle extract directory for TwitchDownloaderCLI_mac
DOTNET_DIR=~/.dotnet/bundle_extract

# Create the directory if it doesn't exist
if [ ! -d "$DOTNET_DIR" ]; then
    echo "Creating .NET bundle extract directory at $DOTNET_DIR"
    mkdir -p "$DOTNET_DIR"
fi

# Set the environment variable
export DOTNET_BUNDLE_EXTRACT_BASE_DIR="$DOTNET_DIR"
echo "Set DOTNET_BUNDLE_EXTRACT_BASE_DIR to $DOTNET_BUNDLE_EXTRACT_BASE_DIR"

echo "Starting cleanup process..."
python cleanup.py

echo "Running main pipeline with URL: $TWITCH_URL"
# Create a simple Python script to run the raw_pipeline
cat > run_raw_pipeline.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import sys
import logging
from raw_pipeline import process_raw_files

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)

async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_raw_pipeline.py <twitch_url>")
        sys.exit(1)

    url = sys.argv[1]
    try:
        result = await process_raw_files(url)
        print(f"Pipeline completed successfully for video {result['video_id']}")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Make the script executable
chmod +x run_raw_pipeline.py

# Run the raw pipeline
python run_raw_pipeline.py "$TWITCH_URL"

echo "Pipeline execution completed."
