#!/usr/bin/env python3
"""
Update Analysis for Sliding Windows

This script updates the existing analysis files to use sliding windows
instead of paragraphs. It modifies the audio sentiment analysis and
chat analysis to use the sliding window segments.

Usage:
    python update_analysis_for_sliding_windows.py <video_id>
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_audio_sentiment_analysis(video_id):
    """
    Update audio sentiment analysis to use sliding windows

    Args:
        video_id (str): The video ID to process

    Returns:
        bool: True if successful, False otherwise
    """
    # Define file paths
    segments_file = f"Output/Raw/Transcripts/audio_{video_id}_segments.csv"
    sentiment_file = f"Output/Analysis/Audio/audio_{video_id}_sentiment.csv"
    backup_file = f"Output/Analysis/Audio/audio_{video_id}_sentiment_paragraphs.csv"

    # Check if files exist
    if not os.path.exists(segments_file):
        logger.error(f"Segments file not found: {segments_file}")
        return False

    if not os.path.exists(sentiment_file):
        logger.error(f"Sentiment file not found: {sentiment_file}")
        return False

    # Create backup of original sentiment file
    logger.info(f"Creating backup of original sentiment file: {backup_file}")
    shutil.copy2(sentiment_file, backup_file)

    # Load data
    logger.info(f"Loading segments from {segments_file}")
    segments_df = pd.read_csv(segments_file)

    logger.info(f"Loading sentiment data from {sentiment_file}")
    sentiment_df = pd.read_csv(sentiment_file)

    # Create a mapping from original paragraphs to sliding windows
    logger.info("Creating mapping from paragraphs to sliding windows")

    # Initialize a new DataFrame for the updated sentiment analysis
    updated_sentiment = []

    # Process each sliding window
    for _, window in segments_df.iterrows():
        # Find paragraphs that overlap with this window
        overlapping_paragraphs = sentiment_df[
            ((sentiment_df['start_time'] >= window['start_time']) &
             (sentiment_df['start_time'] < window['end_time'])) |
            ((sentiment_df['end_time'] > window['start_time']) &
             (sentiment_df['end_time'] <= window['end_time'])) |
            ((sentiment_df['start_time'] <= window['start_time']) &
             (sentiment_df['end_time'] >= window['end_time']))
        ]

        if overlapping_paragraphs.empty:
            # If no overlapping paragraphs, use default values
            window_sentiment = {
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                'text': window['text'],
                'sentiment_score': 0.0,
                'highlight_score': 0.0,
                'excitement': 0.0,
                'funny': 0.0,
                'happiness': 0.0,
                'anger': 0.0,
                'sadness': 0.0,
                'neutral': 1.0,  # Default to neutral
                'energy_score': 0.0,
                'spectral_centroid': 0.0
            }
        else:
            # Calculate weighted average for each metric based on overlap duration
            total_weight = 0
            weighted_metrics = {
                'sentiment_score': 0.0,
                'highlight_score': 0.0,
                'excitement': 0.0,
                'funny': 0.0,
                'happiness': 0.0,
                'anger': 0.0,
                'sadness': 0.0,
                'neutral': 0.0,
                'energy_score': 0.0,
                'spectral_centroid': 0.0
            }

            for _, paragraph in overlapping_paragraphs.iterrows():
                # Calculate overlap duration
                overlap_start = max(window['start_time'], paragraph['start_time'])
                overlap_end = min(window['end_time'], paragraph['end_time'])
                overlap_duration = overlap_end - overlap_start

                # Skip if no overlap
                if overlap_duration <= 0:
                    continue

                # Use overlap duration as weight
                weight = overlap_duration
                total_weight += weight

                # Weight each metric
                for metric in weighted_metrics.keys():
                    if metric in paragraph:
                        weighted_metrics[metric] += paragraph[metric] * weight

            # Calculate weighted averages
            if total_weight > 0:
                for metric in weighted_metrics:
                    weighted_metrics[metric] /= total_weight

            # Create window sentiment entry
            window_sentiment = {
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                'text': window['text'],
                **weighted_metrics
            }

        updated_sentiment.append(window_sentiment)

    # Create DataFrame from updated sentiment
    updated_sentiment_df = pd.DataFrame(updated_sentiment)

    # Save to CSV
    logger.info(f"Saving updated sentiment analysis to {sentiment_file}")
    updated_sentiment_df.to_csv(sentiment_file, index=False)

    return True

def update_chat_analysis(video_id):
    """
    Update chat analysis to use sliding windows

    Args:
        video_id (str): The video ID to process

    Returns:
        bool: True if successful, False otherwise
    """
    # Define file paths
    segments_file = f"Output/Raw/Transcripts/audio_{video_id}_segments.csv"
    chat_analysis_file = f"Output/Analysis/Chat/{video_id}_highlight_analysis.csv"
    backup_file = f"Output/Analysis/Chat/{video_id}_highlight_analysis_original.csv"

    # Check if files exist
    if not os.path.exists(segments_file):
        logger.error(f"Segments file not found: {segments_file}")
        return False

    if not os.path.exists(chat_analysis_file):
        logger.error(f"Chat analysis file not found: {chat_analysis_file}")
        return False

    # Create backup of original chat analysis file
    logger.info(f"Creating backup of original chat analysis file: {backup_file}")
    shutil.copy2(chat_analysis_file, backup_file)

    # Load data
    logger.info(f"Loading segments from {segments_file}")
    segments_df = pd.read_csv(segments_file)

    logger.info(f"Loading chat analysis data from {chat_analysis_file}")
    chat_df = pd.read_csv(chat_analysis_file)

    # Create a new DataFrame for the updated chat analysis
    updated_chat = []

    # Process each sliding window
    for _, window in segments_df.iterrows():
        # Find chat segments that overlap with this window
        overlapping_chat = chat_df[
            ((chat_df['start_time'] >= window['start_time']) &
             (chat_df['start_time'] < window['end_time'])) |
            ((chat_df['end_time'] > window['start_time']) &
             (chat_df['end_time'] <= window['end_time'])) |
            ((chat_df['start_time'] <= window['start_time']) &
             (chat_df['end_time'] >= window['end_time']))
        ]

        if overlapping_chat.empty:
            # If no overlapping chat segments, use default values
            window_chat = {
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                'message_count': 0,
                'avg_sentiment': 0.0,
                'avg_highlight': 0.0,
                'avg_excitement': 0.0,
                'avg_funny': 0.0,
                'avg_happiness': 0.0,
                'avg_anger': 0.0,
                'avg_sadness': 0.0,
                'avg_neutral': 1.0  # Default to neutral
            }
        else:
            # Calculate weighted average for each metric based on overlap duration
            total_weight = 0
            weighted_metrics = {
                'message_count': 0,
                'avg_sentiment': 0.0,
                'avg_highlight': 0.0,
                'avg_excitement': 0.0,
                'avg_funny': 0.0,
                'avg_happiness': 0.0,
                'avg_anger': 0.0,
                'avg_sadness': 0.0,
                'avg_neutral': 0.0
            }

            for _, chat_segment in overlapping_chat.iterrows():
                # Calculate overlap duration
                overlap_start = max(window['start_time'], chat_segment['start_time'])
                overlap_end = min(window['end_time'], chat_segment['end_time'])
                overlap_duration = overlap_end - overlap_start

                # Skip if no overlap
                if overlap_duration <= 0:
                    continue

                # Use overlap duration as weight
                weight = overlap_duration
                total_weight += weight

                # Weight each metric
                for metric in weighted_metrics.keys():
                    if metric in chat_segment:
                        if metric == 'message_count':
                            # For message count, scale by overlap ratio
                            segment_duration = chat_segment['end_time'] - chat_segment['start_time']
                            if segment_duration > 0:
                                overlap_ratio = overlap_duration / segment_duration
                                weighted_metrics[metric] += chat_segment[metric] * overlap_ratio
                        else:
                            weighted_metrics[metric] += chat_segment[metric] * weight

            # Calculate weighted averages
            if total_weight > 0:
                for metric in weighted_metrics:
                    if metric != 'message_count':  # Don't average message count
                        weighted_metrics[metric] /= total_weight

            # Create window chat entry
            window_chat = {
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                **weighted_metrics
            }

        updated_chat.append(window_chat)

    # Create DataFrame from updated chat analysis
    updated_chat_df = pd.DataFrame(updated_chat)

    # Save to CSV
    logger.info(f"Saving updated chat analysis to {chat_analysis_file}")
    updated_chat_df.to_csv(chat_analysis_file, index=False)

    return True

def main(video_id):
    """
    Main function to update analysis for sliding windows

    Args:
        video_id (str): The video ID to process
    """
    logger.info(f"Starting analysis update for sliding windows for video ID: {video_id}")

    # First, generate sliding windows
    logger.info("Generating sliding windows")
    os.system(f"python sliding_window_generator.py {video_id}")

    # Update audio sentiment analysis
    logger.info("Updating audio sentiment analysis")
    audio_result = update_audio_sentiment_analysis(video_id)

    # Update chat analysis
    logger.info("Updating chat analysis")
    chat_result = update_chat_analysis(video_id)

    if audio_result and chat_result:
        logger.info("Successfully updated analysis for sliding windows")
        return True
    else:
        logger.error("Failed to update analysis for sliding windows")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_analysis_for_sliding_windows.py <video_id>")
        sys.exit(1)

    video_id = sys.argv[1]

    success = main(video_id)

    if success:
        print(f"Successfully updated analysis for sliding windows for video ID: {video_id}")
        sys.exit(0)
    else:
        print(f"Failed to update analysis for sliding windows for video ID: {video_id}")
        sys.exit(1)
