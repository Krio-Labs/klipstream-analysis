#!/usr/bin/env python3
"""
Raw Pipeline Script

This script orchestrates the raw file processing and GCS upload for a Twitch VOD.
It performs the following steps:
1. Downloads the video and extracts audio
2. Generates a transcript from the audio
3. Generates a waveform from the audio
4. Downloads the Twitch chat
5. Uploads all these files to Google Cloud Storage

All raw files are stored in the Output/Raw directory structure and are not deleted.
"""

import asyncio
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import the raw file processor
from raw_file_processor import process_raw_files

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the raw pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process raw files for a Twitch VOD')
    parser.add_argument('url', type=str, help='Twitch VOD URL to process')
    args = parser.parse_args()
    
    try:
        # Process raw files
        logger.info(f"Starting raw pipeline for URL: {args.url}")
        result = await process_raw_files(args.url)
        
        # Print results
        logger.info(f"Raw pipeline completed successfully for video {result['video_id']}")
        logger.info(f"Files saved to Output/Raw directory")
        logger.info(f"Files uploaded to GCS: {len(result['uploaded_files'])}")
        
        # Return success
        return 0
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        return 1

if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    exit(exit_code)
