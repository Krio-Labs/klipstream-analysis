"""
Analysis Pipeline Processor Module

This module orchestrates the analysis of raw files for Twitch VODs.
"""

import os
import concurrent.futures
import pandas as pd
from pathlib import Path

from utils.logging_setup import setup_logger
from utils.config import (
    RAW_TRANSCRIPTS_DIR,
    RAW_AUDIO_DIR,
    RAW_CHAT_DIR,
    ANALYSIS_DIR
)

from .audio.sentiment import analyze_audio_sentiment
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
        # Create analysis directory if it doesn't exist
        os.makedirs(ANALYSIS_DIR, exist_ok=True)

        # Create subdirectories
        audio_analysis_dir = ANALYSIS_DIR / "Audio"
        chat_analysis_dir = ANALYSIS_DIR / "Chat"
        os.makedirs(audio_analysis_dir, exist_ok=True)
        os.makedirs(chat_analysis_dir, exist_ok=True)

        # Check if required raw files exist
        transcript_file = RAW_TRANSCRIPTS_DIR / f"audio_{video_id}_paragraphs.csv"
        audio_file = RAW_AUDIO_DIR / f"audio_{video_id}.wav"
        chat_file = RAW_CHAT_DIR / f"{video_id}_chat.csv"

        if not transcript_file.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_file}")

        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        if not chat_file.exists():
            raise FileNotFoundError(f"Chat file not found: {chat_file}")

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

            # Delete chat_sentiment.csv
            chat_sentiment_file = chat_analysis_dir / f"{video_id}_chat_sentiment.csv"
            if os.path.exists(chat_sentiment_file):
                os.remove(chat_sentiment_file)
                logger.info(f"Deleted intermediate file: {chat_sentiment_file}")
        except Exception as e:
            logger.warning(f"Error cleaning up intermediate files: {str(e)}")

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

            # Analyze audio transcription highlights
            logger.info("Step 5: Analyzing audio transcription highlights")
            audio_highlights_future = executor.submit(
                analyze_transcription_highlights,
                video_id,
                str(transcript_file),
                str(audio_analysis_dir)
            )

            # Plot audio metrics
            logger.info("Step 6: Plotting audio metrics")
            audio_plot_future = executor.submit(
                plot_metrics,
                str(audio_analysis_dir),
                video_id
            )

            # Wait for all remaining tasks to complete
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

        # Return results
        return {
            "status": "completed",
            "video_id": video_id,
            "audio_sentiment": audio_sentiment_result,
            "audio_highlights": audio_highlights_result,
            "audio_plot": audio_plot_result,
            "chat_sentiment": chat_sentiment_result,
            "chat_intervals": chat_intervals_result,
            "chat_highlights": chat_highlights_result,
            "integration": integration_result
        }

    except Exception as e:
        logger.error(f"Error processing analysis: {str(e)}")
        raise
