"""
Processor Module

This module orchestrates the raw file processing for Twitch VODs.
"""

import os
import logging
import asyncio
import concurrent.futures
from pathlib import Path

from utils.config import (
    create_directories,
    RAW_VIDEOS_DIR,
    RAW_AUDIO_DIR,
    RAW_TRANSCRIPTS_DIR,
    RAW_WAVEFORMS_DIR,
    RAW_CHAT_DIR
)
from utils.logging_setup import setup_logger
from utils.helpers import extract_video_id

from .downloader import TwitchVideoDownloader
from .transcriber import TranscriptionHandler
from .waveform import generate_waveform
from .chat import download_chat_data
from .uploader import upload_to_gcs

# Set up logger
logger = setup_logger("processor", "raw_processor.log")

async def process_raw_files(url):
    """
    Process raw files for a Twitch VOD
    
    This function:
    1. Downloads the video and extracts audio
    2. Generates a transcript from the audio
    3. Generates a waveform from the audio
    4. Downloads the Twitch chat
    5. Uploads all these files to Google Cloud Storage
    
    Args:
        url (str): Twitch VOD URL
        
    Returns:
        dict: Dictionary with results and metadata
    """
    try:
        # Create directory structure
        create_directories()
        
        # Download video and audio
        downloader = TwitchVideoDownloader()
        download_result = await downloader.process_video(url)
        video_id = download_result["video_id"]
        
        # Create a dictionary to store all file paths
        files = {
            "video_id": video_id,
            "video_file": download_result["video_file"],
            "audio_file": download_result["audio_file"]
        }
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start chat download in parallel
            chat_future = executor.submit(download_chat_data, video_id)
            
            # Generate transcript (depends on audio)
            transcriber = TranscriptionHandler()
            transcript_result = await transcriber.process_audio_files(
                video_id, 
                str(download_result["audio_file"])
            )
            files.update(transcript_result)
            
            # Generate waveform (depends on audio)
            waveform_future = executor.submit(
                generate_waveform, 
                video_id, 
                str(download_result["audio_file"])
            )
            
            # Wait for parallel tasks to complete
            chat_result = chat_future.result()
            waveform_result = waveform_future.result()
            
            # Update files dictionary
            files.update(chat_result)
            files.update(waveform_result)
        
        # Upload files to GCS
        uploaded_files = upload_to_gcs(video_id, files)
        
        # Return results
        return {
            "status": "completed",
            "video_id": video_id,
            "twitch_info": download_result.get("twitch_info", {}),
            "files": files,
            "uploaded_files": uploaded_files
        }
    
    except Exception as e:
        logger.error(f"Error processing raw files: {str(e)}")
        raise
