#!/usr/bin/env python3
"""
Audio Utilities for KlipStream Analysis

This module provides audio processing utilities for the transcription pipeline,
including duration calculation, format conversion, and audio validation.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict
from pydub import AudioSegment

from utils.logging_setup import setup_logger

logger = setup_logger("audio_utils", "audio_utils.log")

def get_audio_duration(audio_file_path: str) -> float:
    """
    Get the duration of an audio file in seconds
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        float: Duration in seconds, or 0.0 if unable to determine
    """
    try:
        audio = AudioSegment.from_file(audio_file_path)
        duration_seconds = len(audio) / 1000.0
        logger.info(f"Audio duration: {duration_seconds:.2f} seconds")
        return duration_seconds
    except Exception as e:
        logger.error(f"Failed to get audio duration for {audio_file_path}: {e}")
        return 0.0

def get_audio_info(audio_file_path: str) -> Dict:
    """
    Get comprehensive audio file information
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        Dict: Audio information including duration, format, channels, etc.
    """
    try:
        audio = AudioSegment.from_file(audio_file_path)
        file_size = os.path.getsize(audio_file_path)
        
        info = {
            "duration_seconds": len(audio) / 1000.0,
            "duration_minutes": len(audio) / 60000.0,
            "duration_hours": len(audio) / 3600000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "format": Path(audio_file_path).suffix.lower(),
            "bitrate": getattr(audio, 'bitrate', None)
        }
        
        logger.info(f"Audio info: {info['duration_minutes']:.1f}min, {info['sample_rate']}Hz, {info['channels']}ch")
        return info
        
    except Exception as e:
        logger.error(f"Failed to get audio info for {audio_file_path}: {e}")
        return {}

def convert_audio_format(input_path: str, output_path: str, target_format: str = "wav") -> bool:
    """
    Convert audio file to target format
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path for output audio file
        target_format (str): Target format (wav, mp3, flac, etc.)
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        audio = AudioSegment.from_file(input_path)
        
        # Optimize for transcription (mono, 16kHz)
        if target_format.lower() == "wav":
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set to 16kHz
        
        audio.export(output_path, format=target_format)
        logger.info(f"Converted {input_path} to {output_path} ({target_format})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {input_path} to {target_format}: {e}")
        return False

def validate_audio_file(audio_file_path: str) -> Tuple[bool, str]:
    """
    Validate audio file for transcription processing
    
    Args:
        audio_file_path (str): Path to audio file
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(audio_file_path):
            return False, "Audio file does not exist"
        
        # Check file size
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            return False, "Audio file is empty"
        
        if file_size > 2 * 1024 * 1024 * 1024:  # 2GB limit
            return False, "Audio file too large (>2GB)"
        
        # Try to load audio
        audio = AudioSegment.from_file(audio_file_path)
        
        # Check duration
        duration_seconds = len(audio) / 1000.0
        if duration_seconds < 1:
            return False, "Audio file too short (<1 second)"
        
        if duration_seconds > 12 * 3600:  # 12 hours limit
            return False, "Audio file too long (>12 hours)"
        
        # Check audio properties
        if audio.frame_rate < 8000:
            return False, "Sample rate too low (<8kHz)"
        
        logger.info(f"Audio file validation passed: {duration_seconds:.1f}s, {audio.frame_rate}Hz")
        return True, "Valid audio file"
        
    except Exception as e:
        error_msg = f"Audio validation failed: {e}"
        logger.error(error_msg)
        return False, error_msg

def optimize_audio_for_transcription(input_path: str, output_path: str) -> bool:
    """
    Optimize audio file for transcription processing
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path for optimized output file
        
    Returns:
        bool: True if optimization successful, False otherwise
    """
    try:
        audio = AudioSegment.from_file(input_path)
        
        # Optimization for transcription
        optimized_audio = audio.set_channels(1)  # Convert to mono
        optimized_audio = optimized_audio.set_frame_rate(16000)  # 16kHz sample rate
        
        # Normalize audio levels
        optimized_audio = optimized_audio.normalize()
        
        # Export as WAV for best compatibility
        optimized_audio.export(output_path, format="wav")
        
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        optimized_size = os.path.getsize(output_path) / (1024 * 1024)
        
        logger.info(f"Audio optimized: {original_size:.1f}MB → {optimized_size:.1f}MB")
        return True
        
    except Exception as e:
        logger.error(f"Failed to optimize audio {input_path}: {e}")
        return False

def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """
    Extract audio from video file using ffmpeg
    
    Args:
        video_path (str): Path to video file
        audio_path (str): Path for extracted audio file
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',  # MP3 codec
            '-ab', '128k',  # 128kbps bitrate
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Audio extracted from {video_path} to {audio_path}")
            return True
        else:
            logger.error(f"ffmpeg failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to extract audio from {video_path}: {e}")
        return False

def calculate_audio_processing_cost(duration_seconds: float, method: str = "deepgram") -> float:
    """
    Calculate estimated cost for audio processing
    
    Args:
        duration_seconds (float): Audio duration in seconds
        method (str): Processing method (deepgram, parakeet_gpu, parakeet_cpu)
        
    Returns:
        float: Estimated cost in USD
    """
    duration_minutes = duration_seconds / 60.0
    
    if method == "deepgram":
        return duration_minutes * 0.0045  # $0.0045 per minute
    elif method == "parakeet_gpu":
        # GPU processing time + GPU cost
        processing_time_hours = duration_seconds / (40 * 3600)  # 40x real-time
        return processing_time_hours * 0.45  # $0.45 per GPU hour
    elif method == "parakeet_cpu":
        # CPU processing time + CPU cost
        processing_time_hours = duration_seconds / (2 * 3600)  # 2x real-time
        return processing_time_hours * 0.10  # $0.10 per CPU hour
    else:
        return 0.0

def estimate_processing_time(duration_seconds: float, method: str = "parakeet_gpu") -> float:
    """
    Estimate processing time for audio transcription
    
    Args:
        duration_seconds (float): Audio duration in seconds
        method (str): Processing method
        
    Returns:
        float: Estimated processing time in seconds
    """
    if method == "deepgram":
        return 10.0  # ~10 seconds regardless of duration
    elif method == "parakeet_gpu":
        return duration_seconds / 40.0  # 40x real-time
    elif method == "parakeet_cpu":
        return duration_seconds / 2.0  # 2x real-time
    else:
        return duration_seconds  # 1x real-time fallback

# Audio format compatibility
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma']

def is_supported_audio_format(file_path: str) -> bool:
    """Check if audio format is supported"""
    return Path(file_path).suffix.lower() in SUPPORTED_AUDIO_FORMATS

def get_recommended_transcription_method(duration_seconds: float, gpu_available: bool = True) -> str:
    """
    Get recommended transcription method based on audio duration and resources
    
    Args:
        duration_seconds (float): Audio duration in seconds
        gpu_available (bool): Whether GPU is available
        
    Returns:
        str: Recommended method (parakeet_gpu, parakeet_cpu, deepgram, hybrid)
    """
    duration_hours = duration_seconds / 3600.0
    
    if not gpu_available:
        if duration_hours < 2:
            return "parakeet_cpu"
        else:
            return "deepgram"
    
    # GPU available
    if duration_hours < 2:
        return "parakeet_gpu"
    elif duration_hours < 4:
        return "hybrid"
    else:
        return "deepgram"
