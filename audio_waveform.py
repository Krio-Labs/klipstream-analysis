from pydub import AudioSegment
import numpy as np
import math
from typing import List
import json
import os
from pathlib import Path

def process_audio_file(samples_size: int = 20000, duration: float = None) -> List[float]:    
    """
    Process an audio file from local outputs folder and return normalized amplitude data.
    """
    # Get local outputs folder path
    outputs_dir = "outputs"
    
    # Find first MP3 file in outputs folder
    mp3_files = list(Path(outputs_dir).glob("*.mp3"))
    if not mp3_files:
        raise FileNotFoundError(f"No MP3 files found in {outputs_dir}")
    
    file_path = str(mp3_files[0])
    print(f"Processing: {file_path}")
    
    # Load audio file using pydub
    audio = AudioSegment.from_mp3(file_path)
    
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

if __name__ == "__main__":
    waveform_data = process_audio_file()
    print(len(waveform_data), "data points generated")
    
    # Save to outputs folder with fixed filename
    output_file = str(Path("outputs") / "audio_waveform.json")
    
    with open(output_file, 'w') as f:
        json.dump(waveform_data, f)
    print(f"File saved successfully. Size: {Path(output_file).stat().st_size / 1024:.2f} KB")