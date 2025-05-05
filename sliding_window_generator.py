#!/usr/bin/env python3
"""
Sliding Window Generator

This script creates sliding windows from transcript data for more effective
highlight detection. It uses a sliding window approach with configurable
window size and overlap.

Usage:
    python sliding_window_generator.py <video_id> [window_size] [overlap]
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_sliding_windows(video_id, window_size=60, overlap=30):
    """
    Generate sliding windows from transcript data

    Args:
        video_id (str): The video ID to process
        window_size (int): Size of each window in seconds
        overlap (int): Overlap between consecutive windows in seconds

    Returns:
        bool: True if successful, False otherwise
    """
    # Define file paths
    words_file = f"Output/Raw/Transcripts/audio_{video_id}_words.csv"
    paragraphs_file = f"Output/Raw/Transcripts/audio_{video_id}_paragraphs.csv"
    output_dir = "Output/Analysis/Segments"
    output_file = f"{output_dir}/{video_id}_sliding_windows.csv"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if files exist
    if not os.path.exists(words_file):
        logger.error(f"Words file not found: {words_file}")
        return False

    if not os.path.exists(paragraphs_file):
        logger.error(f"Paragraphs file not found: {paragraphs_file}")
        return False

    try:
        # Load transcript data
        logger.info(f"Loading words from {words_file}")
        words_df = pd.read_csv(words_file)

        logger.info(f"Loading paragraphs from {paragraphs_file}")
        paragraphs_df = pd.read_csv(paragraphs_file)

        # Validate data
        if words_df.empty or paragraphs_df.empty:
            logger.error("Empty transcript data")
            return False

        # Check for required columns in words file
        words_required_cols = ['start_time', 'end_time', 'word']
        for col in words_required_cols:
            if col not in words_df.columns:
                logger.error(f"Missing required column in words file: {col}")
                return False

        # Check for required columns in paragraphs file
        paragraphs_required_cols = ['start_time', 'end_time', 'text']
        for col in paragraphs_required_cols:
            if col not in paragraphs_df.columns:
                logger.error(f"Missing required column in paragraphs file: {col}")
                return False

        # Get the total duration of the audio
        # Always start at 0 regardless of the actual first word timestamp
        min_time = 0
        max_time = max(words_df['end_time'].max(), paragraphs_df['end_time'].max())
        total_duration = max_time

        logger.info(f"Total audio duration: {total_duration:.2f} seconds")
        logger.info(f"Time range: {min_time} - {max_time:.2f}")

        # Create sliding windows
        step_size = window_size - overlap

        # Generate window boundaries
        windows = []
        window_id = 0

        for start_time in np.arange(min_time, max_time - window_size + 1, step_size):
            end_time = start_time + window_size

            # Ensure end time doesn't exceed max time
            if end_time > max_time:
                end_time = max_time

            # Skip if window is too small
            if end_time - start_time < window_size * 0.5:
                continue

            # Get words in this window - all words that are fully contained or partially overlap
            window_words = words_df[
                # Words fully contained in the window
                ((words_df['start_time'] >= start_time) & (words_df['end_time'] <= end_time)) |
                # Words that start before but end within the window
                ((words_df['start_time'] < start_time) & (words_df['end_time'] > start_time) & (words_df['end_time'] <= end_time)) |
                # Words that start within but end after the window
                ((words_df['start_time'] >= start_time) & (words_df['start_time'] < end_time) & (words_df['end_time'] > end_time)) |
                # Words that completely span the window
                ((words_df['start_time'] <= start_time) & (words_df['end_time'] >= end_time))
            ]

            # Get paragraphs that overlap with this window
            window_paragraphs = paragraphs_df[
                (paragraphs_df['start_time'] < end_time) &
                (paragraphs_df['end_time'] > start_time)
            ]

            # Combine text from all paragraphs in the window
            window_text = ""
            for _, paragraph in window_paragraphs.iterrows():
                # Calculate overlap
                overlap_start = max(start_time, paragraph['start_time'])
                overlap_end = min(end_time, paragraph['end_time'])
                overlap_duration = overlap_end - overlap_start
                paragraph_duration = paragraph['end_time'] - paragraph['start_time']

                # Only include if significant overlap
                if paragraph_duration > 0 and overlap_duration / paragraph_duration >= 0.3:
                    if window_text:
                        window_text += " "
                    window_text += paragraph['text']

            # Count the words that are primarily in this window
            word_count = 0
            for _, word in window_words.iterrows():
                # Calculate how much of the word is in this window
                word_start = word['start_time']
                word_end = word['end_time']
                word_duration = word_end - word_start

                if word_duration <= 0:
                    # Skip invalid words
                    continue

                # Calculate overlap with window
                overlap_start = max(start_time, word_start)
                overlap_end = min(end_time, word_end)
                overlap_duration = overlap_end - overlap_start

                # Calculate what percentage of the word is in this window
                overlap_ratio = overlap_duration / word_duration

                # Count word if majority is in this window
                if overlap_ratio >= 0.5:
                    word_count += 1

            # Calculate speech rate (words per second)
            speech_rate = word_count / window_size if window_size > 0 else 0

            # Create window entry
            window_entry = {
                'window_id': window_id,
                'start_time': start_time,
                'end_time': end_time,
                'word_count': word_count,
                'speech_rate': speech_rate,
                'text': window_text
            }

            windows.append(window_entry)
            window_id += 1

        # Create DataFrame from windows
        windows_df = pd.DataFrame(windows)

        # Validate windows
        if windows_df.empty:
            logger.error("No valid windows generated")
            return False

        # Save sliding windows to segments file
        segments_file = f"Output/Raw/Transcripts/audio_{video_id}_segments.csv"
        logger.info(f"Saving {len(windows_df)} sliding windows to segments file: {segments_file}")
        windows_df.to_csv(segments_file, index=False)

        # Also save to the sliding windows file for reference
        logger.info(f"Also saving to sliding windows file: {output_file}")
        windows_df.to_csv(output_file, index=False)

        logger.info("Sliding window generation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error generating sliding windows: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sliding_window_generator.py <video_id> [window_size] [overlap]")
        sys.exit(1)

    video_id = sys.argv[1]

    window_size = 60
    overlap = 30

    if len(sys.argv) >= 3:
        window_size = int(sys.argv[2])

    if len(sys.argv) >= 4:
        overlap = int(sys.argv[3])

    success = generate_sliding_windows(video_id, window_size, overlap)

    if success:
        print(f"Successfully generated sliding windows for video ID: {video_id}")
        sys.exit(0)
    else:
        print(f"Failed to generate sliding windows for video ID: {video_id}")
        sys.exit(1)
