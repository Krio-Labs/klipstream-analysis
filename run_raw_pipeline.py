#!/usr/bin/env python3
import asyncio
import sys
import logging
import traceback
from raw_pipeline import process_raw_files
from utils.logging_setup import setup_logger

# Set up logger for this script only
logger = setup_logger("raw_pipeline_runner", "main.log")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_raw_pipeline.py <twitch_url>")
        sys.exit(1)

    url = sys.argv[1]
    try:
        result = await process_raw_files(url)
        logger.info(f"Pipeline completed successfully for video {result['video_id']}")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())
