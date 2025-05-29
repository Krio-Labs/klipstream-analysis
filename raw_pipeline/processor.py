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
from convex_integration import STATUS_QUEUED, STATUS_DOWNLOADING, STATUS_FETCHING_CHAT, STATUS_TRANSCRIBING, STATUS_ANALYZING, STATUS_FINDING_HIGHLIGHTS, STATUS_COMPLETED, STATUS_FAILED
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
    3. Generates a transcript from the audio
    4. Generates a waveform from the audio
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
        logger.info("Cleaning up directories from previous runs...")
        cleanup_directories()

        # Create directory structure (already done by cleanup)
        logger.info("Directory structure created")

        # Extract video ID from URL
        video_id = extract_video_id(url)

        # Update Convex status to "Downloading"
        logger.info(f"Updating Convex status to '{STATUS_DOWNLOADING}' for video ID: {video_id}")
        convex_manager.update_video_status(video_id, STATUS_DOWNLOADING)

        # Start video download and chat download in parallel
        logger.info("Starting video download and chat download in parallel...")

        # Update Convex status to "Fetching chat" alongside downloading
        logger.info(f"Updating Convex status to '{STATUS_FETCHING_CHAT}' for video ID: {video_id}")
        convex_manager.update_video_status(video_id, STATUS_FETCHING_CHAT)

        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start video download (async)
            downloader = TwitchVideoDownloader()
            download_future = asyncio.create_task(downloader.process_video(url))

            # Start chat download (sync) in a separate thread
            logger.info(f"Downloading chat data for video ID: {video_id}")
            chat_future = executor.submit(download_chat_data, video_id)

            # Wait for both tasks to complete
            download_result = await download_future
            chat_result = chat_future.result()

            logger.info(f"Video download and chat download completed for video ID: {video_id}")

        # Create a dictionary to store all file paths
        files = {
            "video_id": video_id,
            "video_file": download_result["video_file"],
            "audio_file": download_result["audio_file"]
        }

        # Add chat results to files dictionary
        files.update(chat_result)

        # Update Convex status to "Transcribing"
        logger.info(f"Updating Convex status to '{STATUS_TRANSCRIBING}' for video ID: {video_id}")
        convex_manager.update_video_status(video_id, STATUS_TRANSCRIBING)

        # Generate transcript (depends on audio)
        transcriber = TranscriptionHandler()
        transcript_result = await transcriber.process_audio_files(
            video_id,
            str(download_result["audio_file"])
        )
        files.update(transcript_result)

        # Use ThreadPoolExecutor for remaining parallel tasks
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Generate sliding windows from transcript data
            logger.info(f"Generating sliding windows for video ID: {video_id}")
            sliding_window_future = executor.submit(
                generate_sliding_windows,
                video_id
            )

            # Generate waveform (depends on audio)
            waveform_future = executor.submit(
                generate_waveform,
                video_id,
                str(download_result["audio_file"])
            )

            # Wait for parallel tasks to complete
            sliding_window_result = sliding_window_future.result()
            waveform_result = waveform_future.result()

            # Update files dictionary
            files.update(waveform_result)

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
