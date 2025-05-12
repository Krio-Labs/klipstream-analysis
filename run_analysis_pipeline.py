#!/usr/bin/env python3
"""
Run Analysis Pipeline Script

This script runs the analysis pipeline for a Twitch VOD.
"""

import os
import sys
import asyncio
import argparse

from utils.logging_setup import setup_logger
from analysis_pipeline import process_analysis
import analysis_pipeline.processor

# Set up logger
logger = setup_logger("analysis_pipeline", "analysis_pipeline.log")

async def main():
    """
    Main function to run the analysis pipeline

    Usage: python run_analysis_pipeline.py <video_id> [--concurrency N] [--timeout N]
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run analysis pipeline for a Twitch VOD")
    parser.add_argument("video_id", help="Twitch VOD ID")
    parser.add_argument("--concurrency", type=int, help="Maximum number of concurrent API requests")
    parser.add_argument("--timeout", type=int, help="API request timeout in seconds")

    args = parser.parse_args()

    video_id = args.video_id
    concurrency = args.concurrency
    timeout = args.timeout

    # Create output directories if they don't exist
    os.makedirs("output/Analysis", exist_ok=True)
    os.makedirs("output/Analysis/Audio", exist_ok=True)
    os.makedirs("output/Analysis/Chat", exist_ok=True)

    try:
        # Run the analysis pipeline
        logger.info(f"Starting analysis pipeline for video ID: {video_id}")
        logger.info(f"Using concurrency: {concurrency or 'default'}, timeout: {timeout or 'default'}")

        # Store the original function
        original_analyze_audio_sentiment = analysis_pipeline.processor.analyze_audio_sentiment

        # Define a patched version that uses our parameters
        def patched_analyze_audio_sentiment(*args, **kwargs):
            # Override max_concurrent and timeout if provided
            if concurrency is not None:
                kwargs['max_concurrent'] = concurrency
            if timeout is not None:
                kwargs['timeout'] = timeout
            return original_analyze_audio_sentiment(*args, **kwargs)

        # Replace the function with our patched version
        analysis_pipeline.processor.analyze_audio_sentiment = patched_analyze_audio_sentiment

        # Run the analysis
        result = await process_analysis(video_id)

        # Restore the original function
        analysis_pipeline.processor.analyze_audio_sentiment = original_analyze_audio_sentiment

        if result and result.get("status") == "completed":
            logger.info(f"Analysis pipeline completed successfully for video ID: {video_id}")
            print(f"Analysis pipeline completed successfully for video ID: {video_id}")
            print("Results saved to output/Analysis directory")
        else:
            logger.error(f"Analysis pipeline failed for video ID: {video_id}")
            print(f"Analysis pipeline failed for video ID: {video_id}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error running analysis pipeline: {str(e)}")
        print(f"Error running analysis pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
