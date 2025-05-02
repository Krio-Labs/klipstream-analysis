"""
Waveform Module

This module handles generating waveform data from audio files.
"""

import numpy as np
from pydub import AudioSegment
import json
from pathlib import Path
import os

from utils.config import (
    RAW_AUDIO_DIR,
    RAW_WAVEFORMS_DIR
)
from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("waveform")

def process_audio_file(video_id, audio_file_path=None, samples_size=20000):
    """
    Process an audio file for a specific video ID and return normalized amplitude data.
    
    Args:
        video_id (str): The ID of the video to process
        audio_file_path (str, optional): Path to the audio file. If not provided, will search in standard locations.
        samples_size (int): Number of samples to generate for the waveform
        
    Returns:
        list: List of normalized amplitude values
    """
    # Determine audio file path
    if audio_file_path and os.path.exists(audio_file_path):
        file_path = Path(audio_file_path)
    else:
        # Try different locations in order of preference
        possible_audio_paths = [
            RAW_AUDIO_DIR / f"audio_{video_id}.wav",
            Path(f"outputs/audio_{video_id}.wav"),
            Path(f"/tmp/outputs/audio_{video_id}.wav")
        ]
        
        for path in possible_audio_paths:
            if path.exists():
                file_path = path
                break
        else:
            logger.error(f"Audio file not found for video ID: {video_id}")
            return None
    
    logger.info(f"Processing audio file: {file_path}")
    
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_wav(str(file_path))
        
        # Convert to numpy array more efficiently
        samples = np.frombuffer(audio.raw_data, dtype=np.int16)
        if audio.channels == 2:
            samples = samples[::2]  # Take only left channel if stereo
        
        # Calculate block size for visualization
        total_samples = len(samples)
        block_size = max(1, total_samples // samples_size)
        
        # Use numpy operations instead of loops
        blocks = samples[:block_size * samples_size].reshape(-1, block_size)
        filtered_data = np.abs(blocks).mean(axis=1)
        
        # Normalize using numpy operations
        if len(filtered_data) > 0:
            normalized_data = filtered_data / filtered_data.max()
            return normalized_data.tolist()
        
        return []
    
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        return None

def generate_waveform(video_id, audio_file=None):
    """
    Generate waveform data and save it to a file
    
    Args:
        video_id (str): The ID of the video to process
        audio_file (str, optional): Path to the audio file. If not provided, will search in standard locations.
        
    Returns:
        dict: Dictionary with waveform_file path
    """
    try:
        logger.info(f"Generating waveform for {video_id}...")
        
        # Process audio file to generate waveform data
        waveform_data = process_audio_file(video_id, audio_file)
        
        if not waveform_data:
            raise RuntimeError("Failed to generate waveform data")
        
        # Save waveform data to file
        waveform_file = RAW_WAVEFORMS_DIR / f"audio_{video_id}_waveform.json"
        with open(waveform_file, 'w') as f:
            json.dump(waveform_data, f)
        
        logger.info(f"Waveform data saved to: {waveform_file}")
        
        return {
            "waveform_file": waveform_file
        }
    except Exception as e:
        logger.error(f"Error generating waveform: {str(e)}")
        raise
