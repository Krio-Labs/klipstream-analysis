"""
Processor Module

This module orchestrates the raw file processing for Twitch VODs.
"""

import shutil
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
STATUS_GENERATING_WAVEFORM = "Generating waveform"
STATUS_TRANSCRIBING = "Transcribing"
STATUS_FETCHING_CHAT = "Fetching chat"
STATUS_ANALYZING = "Analyzing"
STATUS_FINDING_HIGHLIGHTS = "Finding highlights"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"
from utils.helpers import extract_video_id

from .downloader import TwitchVideoDownloader
from .transcription.router import TranscriptionRouter
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

        # Download video first (includes audio conversion)
        # Note: Convex status is updated to "Downloading" inside the downloader
        logger.info("📹 Downloading video...")
        downloader = TwitchVideoDownloader()
        download_result = await downloader.process_video(url)
        logger.info("✅ Video download completed")

        # Create a dictionary to store all file paths
        files = {
            "video_id": video_id,
            "video_file": download_result["video_file"],
            "audio_file": download_result["audio_file"]
        }

        # Update Convex status to "Generating waveform" before waveform generation
        print(f"📊 Updating status to 'Generating waveform' for video {video_id}...", flush=True)
        success = convex_manager.update_video_status(video_id, STATUS_GENERATING_WAVEFORM)
        if success:
            print(f"✅ Status updated to 'Generating waveform'", flush=True)
        logger.info("🌊 Generating waveform...")
        waveform_result = generate_waveform(
            video_id,
            str(download_result["audio_file"])
        )
        files.update(waveform_result)
        logger.info("✅ Waveform generation completed")

        # Update Convex status to "Transcribing"
        print(f"📊 Updating status to 'Transcribing' for video {video_id}...", flush=True)
        success = convex_manager.update_video_status(video_id, STATUS_TRANSCRIBING)
        if success:
            print(f"✅ Status updated to 'Transcribing'", flush=True)
        logger.info("🎤 Generating transcript...")

        # Generate transcript using new intelligent transcription router
        try:
            transcriber = TranscriptionRouter()
            transcript_result = await transcriber.transcribe(
                audio_file_path=str(download_result["audio_file"]),
                video_id=video_id,
                output_dir=None  # Will use default output directory
            )

            # Check for transcription errors
            if transcript_result.get("error"):
                logger.error(f"Transcription failed: {transcript_result['error']}")
                raise RuntimeError(f"Transcription failed: {transcript_result['error']}")

            # Extract transcription metadata for cost tracking and performance monitoring
            transcription_metadata = transcript_result.get("transcription_metadata", {})
            if transcription_metadata:
                method_used = transcription_metadata.get("method_used", "unknown")
                cost_estimate = transcription_metadata.get("cost_estimate", 0.0)
                gpu_used = transcription_metadata.get("gpu_used", False)
                processing_time = transcription_metadata.get("processing_time_seconds", 0.0)

                logger.info(f"🎤 Transcription completed using {method_used}")
                logger.info(f"💰 Estimated cost: ${cost_estimate:.3f}")
                logger.info(f"🖥️  GPU acceleration: {'Yes' if gpu_used else 'No'}")
                logger.info(f"⏱️  Processing time: {processing_time:.1f}s")

            files.update(transcript_result)
            logger.info("✅ Transcript generation completed")

        except Exception as e:
            logger.error(f"❌ Transcription failed: {str(e)}")
            # Check if fallback to Deepgram is possible
            logger.info("🔄 Attempting fallback transcription with Deepgram...")
            try:
                # Import the legacy transcriber as fallback
                from .transcriber import TranscriptionHandler
                fallback_transcriber = TranscriptionHandler()
                transcript_result = await fallback_transcriber.process_audio_files(
                    video_id,
                    str(download_result["audio_file"])
                )
                files.update(transcript_result)
                logger.info("✅ Fallback transcription completed")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback transcription also failed: {str(fallback_error)}")
                raise RuntimeError(f"All transcription methods failed. Primary: {str(e)}, Fallback: {str(fallback_error)}")

        # Generate sliding windows from transcript data
        logger.info("📊 Generating sliding windows...")
        sliding_window_result = generate_sliding_windows(video_id)

        # Then download chat (last step)
        print(f"📊 Updating status to 'Fetching chat' for video {video_id}...", flush=True)
        success = convex_manager.update_video_status(video_id, STATUS_FETCHING_CHAT)
        if success:
            print(f"✅ Status updated to 'Fetching chat'", flush=True)
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

        # Log file size summary before cleanup
        log_file_size_summary(video_id, files)

        # Clean up temporary directories
        cleanup_temp_directories()

        logger.info("✅ Raw pipeline completed successfully")

        # Convert Path objects to strings for JSON serialization
        from main import convert_paths_to_strings

        # Return results with transcription metadata
        result = {
            "status": "completed",
            "video_id": video_id,
            "twitch_info": download_result.get("twitch_info", {}),
            "files": convert_paths_to_strings(files),
            "uploaded_files": convert_paths_to_strings(uploaded_files)
        }

        # Add transcription metadata if available
        if 'transcription_metadata' in locals() and transcription_metadata:
            result["transcription_metadata"] = transcription_metadata

        return result

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

def log_file_size_summary(video_id: str, files: dict):
    """
    Log a comprehensive summary of all generated file sizes

    Args:
        video_id (str): The video ID
        files (dict): Dictionary containing file paths
    """
    logger.info("📊 FILE SIZE SUMMARY")
    logger.info("=" * 50)

    total_size_bytes = 0
    file_count = 0

    # Define file categories for better organization
    file_categories = {
        "Video Files": ["video_file"],
        "Audio Files": ["audio_file"],
        "Transcript Files": ["segments_file", "words_file", "paragraphs_file"],
        "Chat Files": ["chat_file", "json_file"],
        "Waveform Files": ["waveform_file"]
    }

    for category, file_keys in file_categories.items():
        category_size = 0
        category_files = []

        for key in file_keys:
            if key in files and files[key]:
                file_path = Path(files[key])
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    category_size += file_size
                    total_size_bytes += file_size
                    file_count += 1
                    category_files.append(f"  • {file_path.name}: {file_size_mb:.1f} MB")

        if category_files:
            category_size_mb = category_size / (1024 * 1024)
            logger.info(f"📁 {category} ({category_size_mb:.1f} MB total):")
            for file_info in category_files:
                logger.info(file_info)

    # Overall summary
    total_size_mb = total_size_bytes / (1024 * 1024)
    total_size_gb = total_size_bytes / (1024 * 1024 * 1024)

    logger.info("=" * 50)
    if total_size_gb >= 1.0:
        logger.info(f"🎯 TOTAL: {file_count} files, {total_size_gb:.2f} GB ({total_size_mb:.1f} MB)")
    else:
        logger.info(f"🎯 TOTAL: {file_count} files, {total_size_mb:.1f} MB")
    logger.info("=" * 50)
