#!/usr/bin/env python3
"""
Generate Sliding Windows for Analysis

This script creates sliding windows from transcript data for more effective
highlight detection. It uses a sliding window approach with configurable
window size and overlap.

Usage:
    python generate_sliding_windows.py <video_id> [window_size] [overlap]
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

def create_sliding_windows(video_id, window_size=60, overlap=30):
    """
    Create sliding windows from transcript data

    Args:
        video_id (str): The video ID to process
        window_size (int): Size of each window in seconds
        overlap (int): Overlap between consecutive windows in seconds

    Returns:
        DataFrame: DataFrame with sliding window segments
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
        return None

    if not os.path.exists(paragraphs_file):
        logger.error(f"Paragraphs file not found: {paragraphs_file}")
        return None

    # Load transcript data
    logger.info(f"Loading words from {words_file}")
    words_df = pd.read_csv(words_file)

    logger.info(f"Loading paragraphs from {paragraphs_file}")
    paragraphs_df = pd.read_csv(paragraphs_file)

    # Get the total duration of the audio
    total_duration = words_df['end_time'].max()
    logger.info(f"Total audio duration: {total_duration:.2f} seconds")

    # Create sliding windows
    step_size = window_size - overlap

    # Calculate number of windows
    num_windows = int((total_duration - window_size) / step_size) + 1
    logger.info(f"Creating approximately {num_windows} sliding windows")

    # Initialize list to store window data
    windows = []

    # Process each window
    current_time = 0.0
    window_id = 0

    # Generate all window start times first
    window_starts = []
    while current_time + window_size <= total_duration:
        window_starts.append(current_time)
        current_time += step_size

    logger.info(f"Generated {len(window_starts)} window start times")

    # Process each window
    for window_id, start_time in enumerate(window_starts):
        end_time = start_time + window_size

        # Ensure end time doesn't exceed total duration
        if end_time > total_duration:
            end_time = total_duration

        # Get words in this window
        window_words = words_df[(words_df['start_time'] >= start_time) &
                               (words_df['end_time'] <= end_time)]

        # Handle words that overlap window boundaries
        # Words that start before the window but end within it
        start_overlap_words = words_df[(words_df['start_time'] < start_time) &
                                      (words_df['end_time'] > start_time) &
                                      (words_df['end_time'] <= end_time)]

        # Words that start within the window but end after it
        end_overlap_words = words_df[(words_df['start_time'] >= start_time) &
                                    (words_df['start_time'] < end_time) &
                                    (words_df['end_time'] > end_time)]

        # Calculate the portion of each overlapping word that falls within the window
        for _, word in start_overlap_words.iterrows():
            overlap_duration = word['end_time'] - start_time
            total_duration = word['end_time'] - word['start_time']
            if total_duration > 0:  # Avoid division by zero
                overlap_ratio = overlap_duration / total_duration
                # Only include the word if a significant portion is in the window
                if overlap_ratio >= 0.5:
                    # Adjust the start time to the window boundary
                    adjusted_word = word.copy()
                    adjusted_word['start_time'] = start_time
                    window_words = pd.concat([window_words, pd.DataFrame([adjusted_word])], ignore_index=True)

        for _, word in end_overlap_words.iterrows():
            overlap_duration = end_time - word['start_time']
            total_duration = word['end_time'] - word['start_time']
            if total_duration > 0:  # Avoid division by zero
                overlap_ratio = overlap_duration / total_duration
                # Only include the word if a significant portion is in the window
                if overlap_ratio >= 0.5:
                    # Adjust the end time to the window boundary
                    adjusted_word = word.copy()
                    adjusted_word['end_time'] = end_time
                    window_words = pd.concat([window_words, pd.DataFrame([adjusted_word])], ignore_index=True)

        # Get paragraphs that overlap with this window
        window_paragraphs = paragraphs_df[(paragraphs_df['start_time'] < end_time) &
                                         (paragraphs_df['end_time'] > start_time)]

        # Combine text from all paragraphs in the window
        window_text = ""
        for _, paragraph in window_paragraphs.iterrows():
            # Calculate the portion of the paragraph that falls within the window
            para_start = max(start_time, paragraph['start_time'])
            para_end = min(end_time, paragraph['end_time'])
            para_duration = para_end - para_start
            total_para_duration = paragraph['end_time'] - paragraph['start_time']

            # Only include text if a significant portion of the paragraph is in the window
            if total_para_duration > 0 and para_duration / total_para_duration >= 0.3:
                if window_text:
                    window_text += " "
                window_text += paragraph['text']

        # Calculate word count and speech rate
        word_count = len(window_words)
        speech_rate = word_count / window_size if window_size > 0 else 0

        # Create window entry
        window_entry = {
            'window_id': window_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'word_count': word_count,
            'speech_rate': speech_rate,
            'text': window_text
        }

        windows.append(window_entry)

    # Create DataFrame from windows
    windows_df = pd.DataFrame(windows)

    # Save to CSV
    logger.info(f"Saving {len(windows_df)} sliding windows to {output_file}")
    windows_df.to_csv(output_file, index=False)

    # Also save in the format expected by the existing analysis modules
    # This will allow the existing modules to use the sliding windows without modification
    audio_output_file = f"Output/Raw/Transcripts/audio_{video_id}_segments.csv"

    # Create a copy with the expected column names
    segments_df = windows_df.copy()

    # All segments should be valid at this point

    # Rename columns to match expected format
    segments_df = segments_df.rename(columns={
        'text': 'text',
        'start_time': 'start_time',
        'end_time': 'end_time'
    })

    # Add any missing columns that might be expected by the analysis modules
    if 'speaker' not in segments_df.columns:
        segments_df['speaker'] = 'SPEAKER_00'

    # Save to CSV in the format expected by the existing analysis modules
    logger.info(f"Saving {len(segments_df)} segments to {audio_output_file}")
    segments_df.to_csv(audio_output_file, index=False)

    return windows_df

def main(video_id, window_size=60, overlap=30):
    """
    Main function to create sliding windows

    Args:
        video_id (str): The video ID to process
        window_size (int): Size of each window in seconds
        overlap (int): Overlap between consecutive windows in seconds
    """
    logger.info(f"Starting sliding window generation for video ID: {video_id}")
    logger.info(f"Window size: {window_size} seconds, Overlap: {overlap} seconds")

    # Create sliding windows
    windows_df = create_sliding_windows(video_id, window_size, overlap)

    if windows_df is not None:
        logger.info(f"Successfully created {len(windows_df)} sliding windows")
        return True
    else:
        logger.error("Failed to create sliding windows")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_sliding_windows.py <video_id> [window_size] [overlap]")
        sys.exit(1)

    video_id = sys.argv[1]

    window_size = 60
    overlap = 30

    if len(sys.argv) >= 3:
        window_size = int(sys.argv[2])

    if len(sys.argv) >= 4:
        overlap = int(sys.argv[3])

    success = main(video_id, window_size, overlap)

    if success:
        print(f"Successfully created sliding windows for video ID: {video_id}")
        sys.exit(0)
    else:
        print(f"Failed to create sliding windows for video ID: {video_id}")
        sys.exit(1)
