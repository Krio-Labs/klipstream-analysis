"""
Processor Module

This module orchestrates the raw file processing for Twitch VODs.
"""

import shutil
import time
import asyncio
import concurrent.futures
from pathlib import Path

# Import sliding_window_generator from the same package
from .sliding_window_generator import generate_sliding_windows
from .cleanup import cleanup_directories

from utils.config import (
    DOWNLOADS_DIR,
    DATA_DIR
)
from utils.logging_setup import setup_logger
from utils.file_manager import FileManager
from utils.convex_client_updated import ConvexManager
# Video processing status constants
STATUS_QUEUED = "Queued"
STATUS_DOWNLOADING = "Downloading"
STATUS_PROCESSING = "Processing"
STATUS_TRANSCRIBING = "Transcribing"
STATUS_FETCHING_CHAT = "Fetching chat"
STATUS_ANALYZING = "Analyzing"
STATUS_FINDING_HIGHLIGHTS = "Finding highlights"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"
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
    1. Cleans up directories from previous runs
    2. Downloads the video and extracts audio
    3. Generates a waveform from the audio
    4. Generates a transcript from the audio
    5. Downloads the Twitch chat
    6. Uploads all these files to Google Cloud Storage

    Args:
        url (str): Twitch VOD URL

    Returns:
        dict: Dictionary with results and metadata
    """
    # Initialize Convex client
    convex_manager = ConvexManager()

    try:
        # Clean up directories from previous runs
        cleanup_directories()

        # Extract video ID from URL
        video_id = extract_video_id(url)

        # Update Convex status to "Downloading" immediately before download starts
        convex_manager.update_video_status(video_id, STATUS_DOWNLOADING)
        logger.info("📹 Downloading video...")

        # Download video first (includes audio conversion)
        downloader = TwitchVideoDownloader()
        download_result = await downloader.process_video(url)
        logger.info("✅ Video download completed")

        # Create a dictionary to store all file paths
        files = {
            "video_id": video_id,
            "video_file": download_result["video_file"],
            "audio_file": download_result["audio_file"]
        }

        # Update Convex status to "Processing" before waveform generation
        convex_manager.update_video_status(video_id, STATUS_PROCESSING)
        logger.info("🌊 Generating waveform...")
        waveform_result = generate_waveform(
            video_id,
            str(download_result["audio_file"])
        )
        files.update(waveform_result)
        logger.info("✅ Waveform generation completed")

        # Update Convex status to "Transcribing"
        convex_manager.update_video_status(video_id, STATUS_TRANSCRIBING)
        logger.info("🎤 Generating transcript...")

        # Generate transcript (depends on audio)
        transcriber = TranscriptionHandler()
        transcript_result = await transcriber.process_audio_files(
            video_id,
            str(download_result["audio_file"])
        )
        files.update(transcript_result)
        logger.info("✅ Transcript generation completed")

        # Generate sliding windows from transcript data
        logger.info("📊 Generating sliding windows...")
        sliding_window_result = generate_sliding_windows(video_id)

        # Then download chat (last step)
        convex_manager.update_video_status(video_id, STATUS_FETCHING_CHAT)
        logger.info("💬 Downloading chat...")
        chat_result = download_chat_data(video_id)
        logger.info("✅ Chat download completed")

        # Add chat results to files dictionary
        files.update(chat_result)

        # Add segments file to files dictionary if sliding window generation was successful
        if sliding_window_result:
            # Use file manager to get the segments file path
            file_manager = FileManager(video_id)
            segments_file = file_manager.get_local_path("segments")

            if segments_file and segments_file.exists():
                files["segments_file"] = segments_file
                logger.info(f"Added segments file to files dictionary: {segments_file}")
            else:
                # Try fallback path for backward compatibility
                fallback_path = Path(f"output/Raw/Transcripts/audio_{video_id}_segments.csv")
                if fallback_path.exists():
                    files["segments_file"] = fallback_path
                    logger.info(f"Added segments file from fallback path to files dictionary: {fallback_path}")
                else:
                    logger.warning(f"Segments file not found after generation")

        # Upload files to GCS
        uploaded_files = upload_to_gcs(video_id, files)

        # Clean up temporary directories
        cleanup_temp_directories()

        # Update Convex status to "Completed" when entire pipeline finishes
        convex_manager.update_video_status(video_id, STATUS_COMPLETED)
        logger.info("✅ Pipeline completed successfully")

        # Convert Path objects to strings for JSON serialization
        from main import convert_paths_to_strings

        # Return results
        return {
            "status": "completed",
            "video_id": video_id,
            "twitch_info": download_result.get("twitch_info", {}),
            "files": convert_paths_to_strings(files),
            "uploaded_files": convert_paths_to_strings(uploaded_files)
        }

    except Exception as e:
        logger.error(f"Error processing raw files: {str(e)}")
        raise

def cleanup_temp_directories():
    """
    Clean up temporary directories (downloads and data) after processing is complete.
    This completely removes the temporary directories.
    """
    try:
        # Clean up downloads directory
        if DOWNLOADS_DIR.exists():
            logger.info(f"Removing downloads directory: {DOWNLOADS_DIR}")
            # Remove the entire directory and all its contents
            shutil.rmtree(DOWNLOADS_DIR)
            logger.info("Downloads directory removed")

        # Clean up data directory
        if DATA_DIR.exists():
            logger.info(f"Removing data directory: {DATA_DIR}")
            # Remove the entire directory and all its contents
            shutil.rmtree(DATA_DIR)
            logger.info("Data directory removed")

    except Exception as e:
        logger.warning(f"Error removing temporary directories: {str(e)}")
        logger.warning("Continuing despite cleanup failure")
