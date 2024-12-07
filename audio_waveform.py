from pydub import AudioSegment
import numpy as np
import math
from typing import List
import json
import os
from pathlib import Path

def process_audio_file(video_id: str, samples_size: int = 20000) -> List[float]:    
    """
    Process an audio file for a specific video ID and return normalized amplitude data.
    """
    # Get specific audio file path
    outputs_dir = "outputs"
    file_path = Path(outputs_dir) / f"audio_{video_id}.wav"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    print(f"Processing: {file_path}")
    
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

if __name__ == "__main__":
    waveform_data = process_audio_file()
    print(len(waveform_data), "data points generated")
    
    # Save to outputs folder with fixed filename
    output_file = str(Path("outputs") / "audio_waveform.json")
    
    with open(output_file, 'w') as f:
        json.dump(waveform_data, f)
    print(f"File saved successfully. Size: {Path(output_file).stat().st_size / 1024:.2f} KB")