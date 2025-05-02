#!/usr/bin/env python3
"""
Run Analysis Pipeline Script

This script runs the analysis pipeline for a Twitch VOD.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

from utils.logging_setup import setup_logger
from analysis_pipeline import process_analysis

# Set up logger
logger = setup_logger("analysis_pipeline", "analysis_pipeline.log")

async def main():
    """
    Main function to run the analysis pipeline
    
    Usage: python run_analysis_pipeline.py <video_id>
    """
    if len(sys.argv) != 2:
        print("Usage: python run_analysis_pipeline.py <video_id>")
        sys.exit(1)
    
    video_id = sys.argv[1]
    
    # Create output directories if they don't exist
    os.makedirs("Output/Analysis", exist_ok=True)
    os.makedirs("Output/Analysis/Audio", exist_ok=True)
    os.makedirs("Output/Analysis/Chat", exist_ok=True)
    
    try:
        # Run the analysis pipeline
        logger.info(f"Starting analysis pipeline for video ID: {video_id}")
        result = await process_analysis(video_id)
        
        if result and result.get("status") == "completed":
            logger.info(f"Analysis pipeline completed successfully for video ID: {video_id}")
            print(f"Analysis pipeline completed successfully for video ID: {video_id}")
            print("Results saved to Output/Analysis directory")
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
