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
from utils.convex_client_updated import ConvexManager
# Video processing status constants
STATUS_QUEUED = "Queued"
STATUS_DOWNLOADING = "Downloading"
STATUS_FETCHING_CHAT = "Fetching chat"
STATUS_TRANSCRIBING = "Transcribing"
STATUS_ANALYZING = "Analyzing"
STATUS_FINDING_HIGHLIGHTS = "Finding highlights"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"

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
    # Initialize Convex client
    convex_manager = ConvexManager()

    try:
        # Note: Convex status is updated to "Analyzing" in main.py before calling this function

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

            # IMPORTANT: DO NOT DELETE chat_sentiment.csv as it's needed for integration
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

                logger.info(f"KEEPING chat sentiment file for integration: {chat_sentiment_file}")
                # Double-check that the file still exists
                if os.path.exists(chat_sentiment_file):
                    logger.info(f"Confirmed chat sentiment file still exists at: {chat_sentiment_file}")
                else:
                    logger.error(f"Chat sentiment file disappeared from: {chat_sentiment_file}")
            else:
                logger.warning(f"Chat sentiment file not found at expected path: {chat_sentiment_file}")
        except Exception as e:
            logger.warning(f"Error handling intermediate files: {str(e)}")

        # Now process audio in parallel (limit workers to prevent resource exhaustion)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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

            # Update Convex status to "Finding highlights"
            print(f"📊 Updating status to 'Finding highlights' for video {video_id}...", flush=True)
            success = convex_manager.update_video_status(video_id, STATUS_FINDING_HIGHLIGHTS)
            if success:
                print(f"✅ Status updated to 'Finding highlights'", flush=True)
            logger.info(f"Updating Convex status to '{STATUS_FINDING_HIGHLIGHTS}' for video ID: {video_id}")

            # Add debugging for highlights analysis with enhanced safety
            def analyze_highlights_with_debug():
                print(f"🔍 Starting SAFE highlights analysis for video {video_id}...", flush=True)
                try:
                    # Use the new safe highlights analysis manager
                    from analysis_pipeline.utils.process_manager import safe_highlights_analysis

                    # Execute with enhanced timeout and safety measures
                    result = safe_highlights_analysis(
                        video_id=video_id,
                        input_file=str(transcript_file),
                        output_dir=str(audio_analysis_dir),
                        timeout=90  # 90 second timeout
                    )

                    if result is not None:
                        print(f"✅ SAFE highlights analysis completed for video {video_id}", flush=True)
                    else:
                        print(f"⚠️ SAFE highlights analysis returned no results for video {video_id}", flush=True)

                    return result

                except Exception as e:
                    print(f"❌ SAFE highlights analysis failed for video {video_id}: {e}", flush=True)
                    logger.error(f"SAFE highlights analysis failed: {str(e)}")
                    return None  # Return None instead of raising to prevent pipeline failure

            audio_highlights_future = executor.submit(analyze_highlights_with_debug)

            # Plot audio metrics
            logger.info("Step 7: Plotting audio metrics")
            audio_plot_future = executor.submit(
                plot_metrics,
                str(audio_analysis_dir),
                video_id
            )

            # Wait for all remaining tasks to complete with timeouts
            try:
                sliding_window_result = sliding_window_future.result(timeout=300)  # 5 minutes
                logger.info("✅ Sliding window analysis completed")
            except concurrent.futures.TimeoutError:
                logger.error("❌ Sliding window analysis timed out")
                sliding_window_result = None

            try:
                print(f"⏳ Waiting for SAFE highlights analysis to complete (timeout: 150 seconds)...", flush=True)
                audio_highlights_result = audio_highlights_future.result(timeout=150)  # 2.5 minutes (buffer for 90s internal timeout)
                print(f"✅ SAFE audio highlights analysis completed", flush=True)
                logger.info("✅ SAFE audio highlights analysis completed")
            except concurrent.futures.TimeoutError:
                print(f"⏰ SAFE audio highlights analysis timed out after 150 seconds", flush=True)
                logger.error("❌ SAFE audio highlights analysis timed out - this should not happen with the new safety measures")

                # Force cleanup of any hanging processes
                try:
                    from analysis_pipeline.utils.process_manager import cleanup_all_processes
                    cleanup_all_processes()
                    print(f"🧹 Forced cleanup of hanging processes completed", flush=True)
                except Exception as cleanup_error:
                    logger.error(f"Error during forced cleanup: {cleanup_error}")

                audio_highlights_result = None
            except Exception as e:
                print(f"❌ SAFE audio highlights analysis failed with error: {e}", flush=True)
                logger.error(f"❌ SAFE audio highlights analysis failed: {e}")
                audio_highlights_result = None

            try:
                # Store the plot result for the return value
                audio_plot_result = audio_plot_future.result(timeout=120)  # 2 minutes
                logger.info("✅ Audio plot generation completed")
            except concurrent.futures.TimeoutError:
                logger.error("❌ Audio plot generation timed out")
                audio_plot_result = None

        # Step 7: Integrate chat and audio analysis
        logger.info("Step 7: Integrating chat and audio analysis")
        try:
            # Use the integrated module instead of subprocess
            integration_result = run_integration(video_id)
            if integration_result:
                logger.info("Chat and audio integration completed successfully")
            else:
                logger.warning("Chat and audio integration failed - continuing with pipeline")
                integration_result = True  # Don't fail the entire pipeline for integration issues
        except Exception as e:
            logger.error(f"Error during chat and audio integration: {str(e)}")
            logger.info("Continuing with pipeline despite integration failure")
            integration_result = True  # Don't fail the entire pipeline for integration issues

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
