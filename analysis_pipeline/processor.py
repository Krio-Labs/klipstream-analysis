"""
Analysis Pipeline Processor Module

This module orchestrates the analysis of raw files for Twitch VODs.
"""

import os
import time
import concurrent.futures
import subprocess
import pandas as pd
from pathlib import Path

from utils.logging_setup import setup_logger
from utils.config import (
    BASE_DIR,
    RAW_TRANSCRIPTS_DIR,
    RAW_AUDIO_DIR,
    RAW_CHAT_DIR,
    ANALYSIS_DIR,
    USE_GCS
)
from utils.file_manager import FileManager

from .audio.sentiment_nebius import analyze_audio_sentiment
from .audio.analysis import analyze_transcription_highlights, plot_metrics
from .chat.processor import process_chat_data
from .chat.sentiment import analyze_chat_sentiment
from .chat.analysis import analyze_chat_highlights
from .integration import run_integration

# Set up logger
logger = setup_logger("analysis_processor", "analysis_processor.log")

async def process_analysis(video_id):
    """
    Process analysis for a Twitch VOD

    This function:
    1. Analyzes audio transcription sentiment
    2. Analyzes audio transcription highlights
    3. Processes chat data
    4. Analyzes chat sentiment
    5. Analyzes chat intervals and highlights

    Args:
        video_id (str): Twitch VOD ID

    Returns:
        dict: Dictionary with results and metadata
    """
    try:
        # Initialize file manager
        file_manager = FileManager(video_id)

        # Create analysis directory if it doesn't exist
        os.makedirs(ANALYSIS_DIR, exist_ok=True)

        # Create subdirectories
        audio_analysis_dir = ANALYSIS_DIR / "Audio"
        chat_analysis_dir = ANALYSIS_DIR / "Chat"
        os.makedirs(audio_analysis_dir, exist_ok=True)
        os.makedirs(chat_analysis_dir, exist_ok=True)

        # Get file paths with automatic download from GCS if needed
        transcript_file = file_manager.get_file_path("segments")
        audio_file = file_manager.get_file_path("audio")
        chat_file = file_manager.get_file_path("chat")

        # Check if required files exist or could be downloaded
        if not transcript_file:
            error_msg = f"Transcript file not found for video ID: {video_id}"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

        if not audio_file:
            error_msg = f"Audio file not found for video ID: {video_id}"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

        if not chat_file:
            error_msg = f"Chat file not found for video ID: {video_id}"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

        logger.info(f"All required files found or downloaded successfully")
        logger.info(f"Transcript file: {transcript_file}")
        logger.info(f"Audio file: {audio_file}")
        logger.info(f"Chat file: {chat_file}")

        # Process chat data first
        logger.info("Step 1: Processing chat data")
        chat_process_result = process_chat_data(
            video_id,
            str(chat_file),
            str(chat_analysis_dir)
        )

        if chat_process_result is None or (isinstance(chat_process_result, pd.DataFrame) and chat_process_result.empty):
            logger.error("Chat processing failed, cannot continue with chat analysis")
            return {"status": "failed", "error": "Chat processing failed"}

        # Then analyze chat sentiment
        logger.info("Step 2: Analyzing chat sentiment")
        chat_sentiment_result = analyze_chat_sentiment(
            video_id,
            str(chat_analysis_dir)
        )

        if chat_sentiment_result is None or (isinstance(chat_sentiment_result, pd.DataFrame) and chat_sentiment_result.empty):
            logger.error("Chat sentiment analysis failed, cannot continue with chat interval analysis")
            return {"status": "failed", "error": "Chat sentiment analysis failed"}

        # Analyze chat highlights
        logger.info("Step 3: Analyzing chat highlights")
        chat_highlights_result = analyze_chat_highlights(
            video_id,
            str(chat_analysis_dir)
        )

        # Set chat_intervals_result to None since we're not using it anymore
        chat_intervals_result = None

        # Clean up intermediate files that are no longer needed
        try:
            # Delete processed_chat.csv
            processed_chat_file = chat_analysis_dir / f"{video_id}_processed_chat.csv"
            if os.path.exists(processed_chat_file):
                os.remove(processed_chat_file)
                logger.info(f"Deleted intermediate file: {processed_chat_file}")

            # DO NOT delete chat_sentiment.csv as it's needed for integration
            chat_sentiment_file = chat_analysis_dir / f"{video_id}_chat_sentiment.csv"
            if os.path.exists(chat_sentiment_file):
                # Upload to GCS if enabled
                if USE_GCS:
                    try:
                        # Upload chat sentiment file to GCS
                        if file_manager.upload_to_gcs("chat_sentiment"):
                            logger.info(f"Successfully uploaded chat sentiment file to GCS")
                        else:
                            logger.warning(f"Failed to upload chat sentiment file to GCS")
                    except Exception as upload_error:
                        logger.warning(f"Error uploading chat sentiment file to GCS: {str(upload_error)}")

                logger.info(f"Keeping chat sentiment file for integration: {chat_sentiment_file}")
            else:
                logger.warning(f"Chat sentiment file not found at expected path: {chat_sentiment_file}")
        except Exception as e:
            logger.warning(f"Error handling intermediate files: {str(e)}")

        # Now process audio in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start audio sentiment analysis
            logger.info("Step 4: Processing audio sentiment analysis")
            audio_sentiment_future = executor.submit(
                analyze_audio_sentiment,
                video_id,
                str(transcript_file),
                str(audio_analysis_dir)
            )

            # Wait for audio sentiment analysis to complete
            audio_sentiment_result = audio_sentiment_future.result()

            # Check if sliding window segments file already exists
            # We already checked this at the beginning, but double-check in case it was deleted
            if file_manager.file_exists_locally("segments"):
                logger.info(f"Sliding window segments file already exists")
                sliding_window_future = executor.submit(lambda: True)  # Return True without running the generator
            else:
                # Run sliding window analysis with the file manager
                logger.info("Step 5: Running sliding window analysis")
                sliding_window_future = executor.submit(
                    lambda: subprocess.run(
                        ["python", "raw_pipeline/sliding_window_generator.py", video_id, "60", "30"],
                        check=True
                    )
                )

                # Add retry logic for sliding window generation
                max_retries = 3
                retry_count = 0
                sliding_window_success = False

                while retry_count < max_retries and not sliding_window_success:
                    try:
                        sliding_window_result = sliding_window_future.result()
                        sliding_window_success = True
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Sliding window generation failed (attempt {retry_count}/{max_retries}): {str(e)}")
                        if retry_count < max_retries:
                            logger.info(f"Retrying sliding window generation in 5 seconds...")
                            time.sleep(5)
                            sliding_window_future = executor.submit(
                                lambda: subprocess.run(
                                    ["python", "raw_pipeline/sliding_window_generator.py", video_id, "60", "30"],
                                    check=True
                                )
                            )
                        else:
                            logger.error(f"All {max_retries} attempts to generate sliding windows failed")
                            # Try to download from GCS as a last resort
                            if USE_GCS and file_manager.download_from_gcs("segments"):
                                logger.info(f"Downloaded segments file from GCS as fallback")
                                sliding_window_success = True
                            else:
                                logger.error("Could not generate or download segments file, analysis may fail")

            # Analyze audio transcription highlights
            logger.info("Step 6: Analyzing audio transcription highlights")
            audio_highlights_future = executor.submit(
                analyze_transcription_highlights,
                video_id,
                str(transcript_file),
                str(audio_analysis_dir)
            )

            # Plot audio metrics
            logger.info("Step 7: Plotting audio metrics")
            audio_plot_future = executor.submit(
                plot_metrics,
                str(audio_analysis_dir),
                video_id
            )

            # Wait for all remaining tasks to complete
            sliding_window_result = sliding_window_future.result()
            audio_highlights_result = audio_highlights_future.result()
            # Store the plot result for the return value
            audio_plot_result = audio_plot_future.result()

        # Step 7: Integrate chat and audio analysis
        logger.info("Step 7: Integrating chat and audio analysis")
        try:
            # Use the integrated module instead of subprocess
            integration_result = run_integration(video_id)
            if integration_result:
                logger.info("Chat and audio integration completed successfully")
            else:
                logger.warning("Chat and audio integration failed")
        except Exception as e:
            logger.error(f"Error during chat and audio integration: {str(e)}")
            integration_result = False
            # Continue with pipeline even if integration fails

        # Convert Path objects to strings for JSON serialization
        from main import convert_paths_to_strings

        # Return results
        return {
            "status": "completed",
            "video_id": video_id,
            "audio_sentiment": convert_paths_to_strings(audio_sentiment_result),
            "sliding_window": convert_paths_to_strings(sliding_window_result),
            "audio_highlights": convert_paths_to_strings(audio_highlights_result),
            "audio_plot": convert_paths_to_strings(audio_plot_result),
            "chat_sentiment": convert_paths_to_strings(chat_sentiment_result),
            "chat_intervals": convert_paths_to_strings(chat_intervals_result),
            "chat_highlights": convert_paths_to_strings(chat_highlights_result),
            "integration": convert_paths_to_strings(integration_result)
        }

    except Exception as e:
        logger.error(f"Error processing analysis: {str(e)}")
        raise
