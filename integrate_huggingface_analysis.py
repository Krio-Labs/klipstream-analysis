#!/usr/bin/env python3
"""
Integration script for Hugging Face transcript analysis.

This script integrates the Hugging Face sentiment and emotion analysis
into the main pipeline by:
1. Running the analysis on the transcript data
2. Updating the highlight detection based on the new analysis
3. Generating additional visualizations

Usage:
python integrate_huggingface_analysis.py --video_id VIDEO_ID
"""

import os
import argparse
import pandas as pd
import numpy as np
import logging
import subprocess
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_huggingface_analysis(video_id):
    """
    Run the Hugging Face analysis on the transcript data.

    Args:
        video_id: The ID of the Twitch VOD

    Returns:
        Path to the output file with analysis results
    """
    logger.info(f"Running Hugging Face analysis for video {video_id}...")

    input_file = f"outputs/audio_{video_id}_paragraphs.csv"
    output_file = f"outputs/audio_{video_id}_paragraphs_hf.csv"

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found")
        return None

    # Check if output file already exists
    if os.path.exists(output_file):
        # Check if all entries have been processed
        try:
            df = pd.read_csv(output_file)
            processed = df['hf_sentiment'].notna().sum()
            total = len(df)

            if processed == total:
                logger.info(f"Analysis already completed. Using existing file: {output_file}")
                return output_file
            else:
                logger.info(f"Found partially processed file ({processed}/{total} entries). Continuing analysis...")
        except Exception as e:
            logger.warning(f"Error checking existing file: {e}. Will run full analysis.")

    # Run the batch analysis script
    try:
        subprocess.run([
            "./run_batch_analysis.sh",
            video_id,
            "0",  # Process all entries
            "100"  # Save checkpoint every 100 entries
        ], check=True)

        logger.info(f"Analysis completed successfully. Results saved to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Hugging Face analysis: {e}")

        # Check if we have a partial result
        if os.path.exists(output_file):
            logger.warning(f"Analysis failed but partial results are available in {output_file}")
            return output_file

        return None

def update_highlight_detection(video_id, analysis_file):
    """
    Update the highlight detection based on the Hugging Face analysis.

    Args:
        video_id: The ID of the Twitch VOD
        analysis_file: Path to the file with analysis results

    Returns:
        Path to the updated highlights file
    """
    logger.info(f"Updating highlight detection for video {video_id}...")

    # Load the analysis results
    try:
        df = pd.read_csv(analysis_file)
        logger.info(f"Loaded {len(df)} entries from {analysis_file}")
    except Exception as e:
        logger.error(f"Error loading analysis results: {e}")
        return None

    # Load existing highlights if available
    highlights_file = f"outputs/audio_{video_id}_highlights.json"
    highlights = []

    if os.path.exists(highlights_file):
        try:
            with open(highlights_file, 'r') as f:
                highlights = json.load(f)
            logger.info(f"Loaded {len(highlights)} existing highlights from {highlights_file}")
        except Exception as e:
            logger.error(f"Error loading existing highlights: {e}")

    # Define emotions of interest for highlights
    positive_emotions = ['joy', 'happiness', 'excitement', 'amusement', 'surprise']
    negative_emotions = ['anger', 'frustration', 'sadness', 'fear', 'disappointment']

    # Find potential highlight moments based on emotions
    emotion_highlights = []

    for idx, row in df.iterrows():
        if pd.isna(row.get('hf_emotion')) or row.get('hf_emotion') == 'error':
            continue

        emotion = row['hf_emotion']
        start_time = row.get('start_time')
        end_time = row.get('end_time')
        text = row.get('text', '')

        if not (pd.isna(start_time) or pd.isna(end_time)):
            highlight_type = None

            if emotion in positive_emotions:
                highlight_type = f"positive_{emotion}"
            elif emotion in negative_emotions:
                highlight_type = f"negative_{emotion}"

            if highlight_type:
                emotion_highlights.append({
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'type': highlight_type,
                    'source': 'huggingface_emotion',
                    'text': text,
                    'emotion': emotion
                })

    logger.info(f"Found {len(emotion_highlights)} potential highlights based on emotions")

    # Merge with existing highlights
    all_highlights = highlights + emotion_highlights

    # Sort by start time
    all_highlights.sort(key=lambda x: x['start_time'])

    # Save the updated highlights
    updated_highlights_file = f"outputs/audio_{video_id}_highlights_hf.json"

    try:
        with open(updated_highlights_file, 'w') as f:
            json.dump(all_highlights, f, indent=2)
        logger.info(f"Saved {len(all_highlights)} highlights to {updated_highlights_file}")
        return updated_highlights_file
    except Exception as e:
        logger.error(f"Error saving updated highlights: {e}")
        return None

def generate_visualizations(analysis_file):
    """
    Generate visualizations of the analysis results.

    Args:
        analysis_file: Path to the file with analysis results

    Returns:
        Path to the directory with visualizations
    """
    logger.info(f"Generating visualizations for {analysis_file}...")

    output_dir = "visualizations"

    # Run the visualization script
    try:
        subprocess.run([
            "python", "visualize_analysis.py",
            "--input_file", analysis_file,
            "--output_dir", output_dir
        ], check=True)

        logger.info(f"Visualizations generated successfully in {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating visualizations: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Integrate Hugging Face transcript analysis into the main pipeline')
    parser.add_argument('--video_id', required=True, help='The ID of the Twitch VOD')

    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    # Run the Hugging Face analysis
    analysis_file = run_huggingface_analysis(args.video_id)

    if analysis_file:
        # Update highlight detection
        highlights_file = update_highlight_detection(args.video_id, analysis_file)

        # Generate visualizations
        visualizations_dir = generate_visualizations(analysis_file)

        if highlights_file and visualizations_dir:
            logger.info(f"Integration completed successfully!")
            logger.info(f"Analysis results: {analysis_file}")
            logger.info(f"Updated highlights: {highlights_file}")
            logger.info(f"Visualizations: {visualizations_dir}")
        else:
            logger.warning("Integration completed with some errors")
    else:
        logger.error("Integration failed")

if __name__ == "__main__":
    main()
