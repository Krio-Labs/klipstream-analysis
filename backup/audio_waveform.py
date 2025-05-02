from pydub import AudioSegment
import numpy as np
import math
from typing import List, Optional
import json
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define directory paths
OUTPUT_DIR = Path("outputs")
RAW_DIR = Path("Output/Raw")
RAW_AUDIO_DIR = RAW_DIR / "Audio"
RAW_WAVEFORMS_DIR = RAW_DIR / "Waveforms"

def process_audio_file(video_id: str, audio_file_path: Optional[str] = None, samples_size: int = 20000) -> List[float]:
    """
    Process an audio file for a specific video ID and return normalized amplitude data.

    Args:
        video_id (str): The ID of the video to process
        audio_file_path (str, optional): Path to the audio file. If not provided, will search in standard locations.
        samples_size (int): Number of samples to generate for the waveform

    Returns:
        List[float]: Normalized amplitude data for waveform visualization
    """
    # Determine the audio file path
    if audio_file_path and os.path.exists(audio_file_path):
        file_path = Path(audio_file_path)
    else:
        # Try different locations in order of preference
        possible_paths = [
            RAW_AUDIO_DIR / f"audio_{video_id}.wav",
            OUTPUT_DIR / f"audio_{video_id}.wav",
            Path(f"outputs/audio_{video_id}.wav"),
            Path(f"/tmp/outputs/audio_{video_id}.wav")
        ]

        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        else:
            raise FileNotFoundError(f"Audio file not found for video ID: {video_id}")

    logger.info(f"Processing audio file: {file_path}")

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

def save_waveform_data(video_id: str, waveform_data: List[float], output_dir: Optional[Path] = None) -> Path:
    """
    Save waveform data to a JSON file

    Args:
        video_id (str): The ID of the video
        waveform_data (List[float]): The waveform data to save
        output_dir (Path, optional): Directory to save the file. Defaults to RAW_WAVEFORMS_DIR.

    Returns:
        Path: Path to the saved file
    """
    # Determine output directory
    if output_dir is None:
        output_dir = RAW_WAVEFORMS_DIR

    # Create directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create output file path
    output_file = output_dir / f"audio_{video_id}_waveform.json"

    # Save data to file
    with open(output_file, 'w') as f:
        json.dump(waveform_data, f)

    logger.info(f"Waveform data saved to: {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")

    return output_file

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process audio waveform')
    parser.add_argument('--video-id', type=str, required=True, help='Video ID to process')
    parser.add_argument('--audio-file', type=str, help='Path to audio file (optional)')
    parser.add_argument('--output-dir', type=str, help='Output directory (optional)')
    args = parser.parse_args()

    # Process audio file
    waveform_data = process_audio_file(args.video_id, args.audio_file)
    logger.info(f"Generated {len(waveform_data)} data points")

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else RAW_WAVEFORMS_DIR

    # Save waveform data
    output_file = save_waveform_data(args.video_id, waveform_data, output_dir)
    logger.info(f"Waveform data saved to: {output_file}")