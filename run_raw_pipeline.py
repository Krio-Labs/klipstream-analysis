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
