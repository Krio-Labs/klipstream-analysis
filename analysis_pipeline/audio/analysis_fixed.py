#!/usr/bin/env python3
"""
Fixed Audio Analysis Module with Robust Timeout and Resource Management

This module handles analysis of audio transcriptions to identify highlights and emotion peaks
with comprehensive fixes for threading, signal handling, and resource management issues.
"""

import pandas as pd
import numpy as np
import os
import librosa
import gc
import threading
import time
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any
import contextlib
import psutil

# Set matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("audio_analysis_fixed", "audio_analysis_fixed.log")

class TimeoutError(Exception):
    """Custom timeout error for highlights analysis"""
    pass

class ResourceManager:
    """Manages system resources and cleanup for audio analysis"""
    
    def __init__(self):
        self.start_memory = psutil.virtual_memory().used
        self.peak_memory = self.start_memory
        self.cleanup_callbacks = []
        
    def register_cleanup(self, callback):
        """Register a cleanup callback"""
        self.cleanup_callbacks.append(callback)
    
    def cleanup_all(self):
        """Execute all cleanup callbacks"""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        gc.collect()
        gc.collect()  # Call twice for thorough cleanup
        
    def check_memory_usage(self):
        """Check current memory usage"""
        current_memory = psutil.virtual_memory().used
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # Log if memory usage is high
        memory_gb = current_memory / (1024**3)
        if memory_gb > 8:  # More than 8GB
            logger.warning(f"High memory usage detected: {memory_gb:.2f}GB")
            
        return current_memory

class ThreadSafeTimeout:
    """Thread-safe timeout mechanism that works in worker threads"""
    
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.is_cancelled = False
        self._lock = threading.Lock()
        
    def start(self):
        """Start the timeout"""
        with self._lock:
            self.start_time = time.time()
            self.is_cancelled = False
    
    def check(self):
        """Check if timeout has been exceeded"""
        with self._lock:
            if self.is_cancelled:
                return False
            if self.start_time is None:
                return False
            return (time.time() - self.start_time) > self.timeout_seconds
    
    def cancel(self):
        """Cancel the timeout"""
        with self._lock:
            self.is_cancelled = True
    
    def get_elapsed(self):
        """Get elapsed time"""
        with self._lock:
            if self.start_time is None:
                return 0
            return time.time() - self.start_time

def create_fallback_highlights_result(video_id: str) -> pd.DataFrame:
    """
    Create a fallback highlights result when analysis fails or times out

    Args:
        video_id (str): Video ID for logging

    Returns:
        pd.DataFrame: Empty highlights DataFrame with correct structure
    """
    logger.info(f"🔄 Creating fallback highlights result for video {video_id}")

    # Create empty DataFrame with expected structure
    fallback_data = {
        'start_time': [],
        'end_time': [],
        'duration': [],
        'weighted_highlight_score': [],
        'highlight_score': [],
        'emotion_intensity': [],
        'sentiment_score': [],
        'audio_intensity': [],
        'text': []
    }

    return pd.DataFrame(fallback_data)

def safe_file_operation(operation, *args, timeout=30, **kwargs):
    """Execute file operation with timeout protection"""
    def target(result_container):
        try:
            result = operation(*args, **kwargs)
            result_container['result'] = result
            result_container['success'] = True
        except Exception as e:
            result_container['error'] = str(e)
            result_container['success'] = False

    result_container = {'success': False}
    thread = threading.Thread(target=target, args=(result_container,))
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.error(f"File operation timed out after {timeout} seconds")
        raise TimeoutError(f"File operation timed out after {timeout} seconds")

    if not result_container['success']:
        error_msg = result_container.get('error', 'Unknown error')
        raise Exception(f"File operation failed: {error_msg}")

    return result_container['result']

def load_audio_with_timeout(audio_path: str, timeout: int = 60) -> tuple:
    """Load audio file with timeout protection"""
    def load_audio():
        # Check file size before loading
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Loading audio file: {audio_path} ({file_size_mb:.1f} MB)")
        
        if file_size_mb > 500:  # If larger than 500MB, use chunked loading
            logger.info("Large audio file detected, using chunked processing")
            y, sr = librosa.load(audio_path, duration=300)  # Load only first 5 minutes
        else:
            y, sr = librosa.load(audio_path)
        
        return y, sr
    
    return safe_file_operation(load_audio, timeout=timeout)

def process_audio_features_safe(y, sr, data_length: int, timeout_checker: ThreadSafeTimeout) -> np.ndarray:
    """Process audio features with timeout checking"""
    try:
        # Check timeout before starting
        if timeout_checker.check():
            raise TimeoutError("Timeout during audio feature processing")
        
        # Calculate audio features with smaller frame sizes for efficiency
        frame_length = min(int(sr * 0.03), 2048)  # 30ms frames or 2048 samples max
        hop_length = min(int(sr * 0.01), 512)     # 10ms hop or 512 samples max
        
        logger.info("Calculating RMS energy...")
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Check timeout after RMS
        if timeout_checker.check():
            raise TimeoutError("Timeout during RMS calculation")
        
        logger.info("Calculating spectral centroid...")
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr,
            n_fft=frame_length,
            hop_length=hop_length
        )[0]
        
        # Check timeout after spectral centroid
        if timeout_checker.check():
            raise TimeoutError("Timeout during spectral centroid calculation")
        
        # Get timestamps for audio features
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        
        # Create dummy time array for interpolation if data is empty
        if data_length == 0:
            return np.array([])
        
        # Create time points for interpolation (assuming 1-second intervals)
        data_times = np.arange(data_length)
        
        # Interpolate to match data length
        rms_interp = np.interp(data_times, times, rms)
        centroid_interp = np.interp(data_times, times, spectral_centroid)
        
        # Check timeout after interpolation
        if timeout_checker.check():
            raise TimeoutError("Timeout during feature interpolation")
        
        # Normalize audio features to [0,1] range
        if rms_interp.max() > rms_interp.min():
            rms_norm = (rms_interp - rms_interp.min()) / (rms_interp.max() - rms_interp.min())
        else:
            rms_norm = np.ones_like(rms_interp) * 0.5
            
        if centroid_interp.max() > centroid_interp.min():
            centroid_norm = (centroid_interp - centroid_interp.min()) / (centroid_interp.max() - centroid_interp.min())
        else:
            centroid_norm = np.ones_like(centroid_interp) * 0.5
        
        # Combine audio features
        audio_intensity = (rms_norm * 0.7 + centroid_norm * 0.3)  # Weighted combination
        
        logger.info(f"Audio features calculated successfully: {len(audio_intensity)} samples")
        return audio_intensity
        
    except Exception as e:
        logger.error(f"Error processing audio features: {e}")
        # Return fallback values
        return np.ones(data_length) * 0.5

def analyze_transcription_highlights_safe(video_id: str, input_file: Optional[str] = None,
                                        output_dir: Optional[str] = None,
                                        timeout_seconds: int = 90) -> Optional[pd.DataFrame]:
    """
    Thread-safe analysis of transcription highlights with robust timeout and resource management

    This function is designed to work correctly in ThreadPoolExecutor environments
    without using signal handlers that only work in the main thread.

    Args:
        video_id (str): The ID of the video to analyze
        input_file (str, optional): Path to the input segments CSV file
        output_dir (str, optional): Directory to save output files
        timeout_seconds (int): Timeout in seconds (default: 90)

    Returns:
        pd.DataFrame: DataFrame containing the top highlights, or None if analysis failed
    """
    # Initialize timeout checker and resource manager
    timeout_checker = ThreadSafeTimeout(timeout_seconds)
    resource_manager = ResourceManager()

    timeout_checker.start()

    try:
        logger.info(f"🔍 Starting highlights analysis for video {video_id} (timeout: {timeout_seconds}s)")

        # Quick timeout check - if we're already close to timeout, return fallback
        if timeout_seconds < 30:
            logger.warning(f"⚠️ Timeout too short ({timeout_seconds}s), returning fallback result")
            return create_fallback_highlights_result(video_id)

        # Import here to avoid circular imports
        from utils.config import BASE_DIR, USE_GCS
        from utils.file_manager import FileManager

        # Check timeout early
        if timeout_checker.check():
            logger.warning("⏰ Timeout before analysis start, returning fallback")
            return create_fallback_highlights_result(video_id)

        # Initialize file manager
        file_manager = FileManager(video_id)
        resource_manager.register_cleanup(lambda: logger.info("FileManager cleanup completed"))

        # Define output directory
        if output_dir is None:
            output_dir = Path('output/Analysis/Audio')
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Check timeout after setup
        if timeout_checker.check():
            logger.warning("⏰ Timeout during setup, returning fallback")
            return create_fallback_highlights_result(video_id)
        
        # Determine input file path with timeout protection
        if input_file is None:
            input_file = file_manager.get_file_path("segments")
            if input_file is None:
                # Try alternative paths
                tmp_path = Path(f'/tmp/output/Raw/Transcripts/audio_{video_id}_segments.csv')
                alt_path = Path(f'output/Raw/Transcripts/audio_{video_id}_segments.csv')
                
                if os.path.exists(tmp_path):
                    input_file = tmp_path
                elif os.path.exists(alt_path):
                    input_file = alt_path
                else:
                    # Try to download from GCS with timeout
                    if USE_GCS:
                        try:
                            if safe_file_operation(file_manager.download_from_gcs, "segments", timeout=30):
                                input_file = file_manager.get_local_path("segments")
                        except Exception as e:
                            logger.warning(f"Failed to download segments from GCS: {e}")
                    
                    if input_file is None:
                        logger.error(f"Could not find segments file for video ID: {video_id}")
                        return None
        else:
            input_file = Path(input_file)
        
        # Check if file exists
        if not os.path.exists(input_file):
            logger.error(f"Required file not found: {input_file}")
            return None
        
        # Check timeout before loading data
        if timeout_checker.check():
            raise TimeoutError("Timeout before loading data")
        
        # Load transcription data with timeout protection
        logger.info(f"📊 Loading transcription data from: {input_file}")

        # Check timeout before loading
        if timeout_checker.check():
            logger.warning("⏰ Timeout before loading data, returning fallback")
            return create_fallback_highlights_result(video_id)

        try:
            data = safe_file_operation(pd.read_csv, input_file, timeout=15)  # Reduced timeout
            data.columns = data.columns.str.strip()
            logger.info(f"📊 Loaded {len(data)} segments for analysis")
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}, returning fallback")
            return create_fallback_highlights_result(video_id)

        # Check timeout after loading data
        if timeout_checker.check():
            logger.warning("⏰ Timeout after loading data, returning fallback")
            return create_fallback_highlights_result(video_id)
        
        # Check for pre-calculated audio metrics from sentiment_nebius.py
        has_nebius_features = all(col in data.columns for col in ['speech_rate', 'absolute_intensity', 'relative_intensity'])

        if has_nebius_features:
            # Use pre-calculated audio metrics
            logger.info("Using pre-calculated audio metrics from sentiment file")
            audio_intensity = data['absolute_intensity'].values
        else:
            # Need to calculate audio features from scratch
            logger.info("Pre-calculated audio metrics not found, calculating from audio file")

            try:
                # Try to get the audio file using the file manager
                audio_path = file_manager.get_file_path("audio")

                if not audio_path or not os.path.exists(audio_path):
                    # Try alternative paths
                    alt_paths = [
                        Path(f'output/Raw/Audio/audio_{video_id}.mp3'),
                        Path(f'output/Raw/Audio/audio_{video_id}.wav'),
                        Path(f'/tmp/output/Raw/Audio/audio_{video_id}.mp3'),
                        Path(f'/tmp/output/Raw/Audio/audio_{video_id}.wav'),
                        Path(f'/tmp/outputs/audio_{video_id}.mp3'),
                        Path(f'/tmp/outputs/audio_{video_id}.wav')
                    ]

                    audio_path = None
                    for path in alt_paths:
                        if os.path.exists(path):
                            audio_path = str(path)
                            break

                    if not audio_path and USE_GCS:
                        # Try to download from GCS
                        try:
                            if safe_file_operation(file_manager.download_from_gcs, "audio", timeout=60):
                                audio_path = file_manager.get_local_path("audio")
                        except Exception as e:
                            logger.warning(f"Failed to download audio from GCS: {e}")

                if audio_path and os.path.exists(audio_path):
                    # Check timeout before audio processing
                    if timeout_checker.check():
                        raise TimeoutError("Timeout before audio processing")

                    # Load audio with timeout protection
                    y, sr = load_audio_with_timeout(audio_path, timeout=60)
                    resource_manager.register_cleanup(lambda: logger.info("Audio data cleanup"))

                    # Process audio features with timeout checking
                    audio_intensity = process_audio_features_safe(y, sr, len(data), timeout_checker)

                    # Clean up audio data immediately
                    del y, sr
                    gc.collect()

                else:
                    logger.warning(f"Could not find audio file for video ID: {video_id}")
                    audio_intensity = np.ones(len(data)) * 0.5  # Fallback

            except Exception as e:
                logger.warning(f"Audio processing failed: {e}")
                audio_intensity = np.ones(len(data)) * 0.5  # Fallback

        # Check timeout before highlight processing
        if timeout_checker.check():
            raise TimeoutError("Timeout before highlight processing")

        # Check for required columns
        required_columns = [
            'start_time', 'end_time', 'text',
            'sentiment_score', 'highlight_score',
            'excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]

        # Add missing columns with default values
        for col in missing_columns:
            if col in ['sentiment_score', 'highlight_score']:
                data[col] = 0.0
            elif col in ['excitement', 'funny', 'happiness', 'anger', 'sadness']:
                data[col] = 0.0
            elif col == 'neutral':
                data[col] = 1.0

        # Calculate emotion intensity
        data['emotion_intensity'] = data[['excitement', 'funny', 'happiness', 'anger', 'sadness']].max(axis=1)

        # Ensure audio_intensity matches data length
        if len(data) != len(audio_intensity):
            logger.warning(f"Length mismatch: data has {len(data)} rows, audio_intensity has {len(audio_intensity)} elements")
            if len(data) < len(audio_intensity):
                audio_intensity = audio_intensity[:len(data)]
            else:
                extension = np.ones(len(data) - len(audio_intensity)) * (audio_intensity[-1] if len(audio_intensity) > 0 else 0.5)
                audio_intensity = np.concatenate([audio_intensity, extension])

        # Calculate weighted highlight score
        if has_nebius_features:
            data['weighted_highlight_score'] = (
                data['highlight_score'] * 0.95 +    # Nebius highlight score (95%)
                audio_intensity * 0.05              # Current audio intensity (5%)
            )
        else:
            data['weighted_highlight_score'] = (
                data['highlight_score'] * 0.5 +      # Base highlight score
                data['emotion_intensity'] * 0.2 +    # Emotion contribution
                abs(data['sentiment_score']) * 0.1 + # Sentiment intensity contribution
                audio_intensity * 0.2                # Audio intensity contribution
            )

        # Check timeout before peak detection
        if timeout_checker.check():
            raise TimeoutError("Timeout before peak detection")

        # Find peaks in weighted highlight score
        if has_nebius_features:
            peaks, properties = find_peaks(
                data['weighted_highlight_score'],
                distance=15, prominence=0.25, height=0.35
            )
        else:
            peaks, properties = find_peaks(
                data['weighted_highlight_score'],
                distance=20, prominence=0.3, height=0.4
            )

        logger.info(f"Found {len(peaks)} highlight peaks")

        # Create DataFrame with peak moments
        if len(peaks) > 0:
            peak_moments = data.iloc[peaks].copy()
            peak_moments['prominence'] = properties['prominences']

            # Add audio intensity for peaks
            valid_indices = [i for i in peaks if i < len(audio_intensity)]
            if len(valid_indices) > 0:
                peak_moments['audio_intensity'] = audio_intensity[peaks]
            else:
                peak_moments['audio_intensity'] = 0.5

            # Sort by weighted_highlight_score
            peak_moments = peak_moments.sort_values(by='weighted_highlight_score', ascending=False)

            # Select top 10 moments
            top_highlights = peak_moments.head(10).copy()

            # Add duration column
            top_highlights['duration'] = top_highlights['end_time'] - top_highlights['start_time']

            # Select output columns
            output_columns = [
                'start_time', 'end_time', 'duration',
                'weighted_highlight_score', 'highlight_score',
                'emotion_intensity', 'sentiment_score', 'audio_intensity',
                'text'
            ]
            top_highlights = top_highlights[output_columns]

            # Save to CSV with timeout protection
            output_file = output_dir / f"audio_{video_id}_top_highlights.csv"
            safe_file_operation(top_highlights.to_csv, output_file, index=False, timeout=30)
            logger.info(f"Top highlights saved to {output_file}")

        else:
            logger.warning("No highlight peaks found")
            top_highlights = pd.DataFrame()

        logger.info(f"Highlights analysis completed for video {video_id} in {timeout_checker.get_elapsed():.1f}s")
        return top_highlights
        
    except TimeoutError as e:
        logger.error(f"Highlights analysis timed out: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in highlights analysis: {e}")
        return None
    finally:
        # Always cleanup resources
        timeout_checker.cancel()
        resource_manager.cleanup_all()
        logger.info("Resource cleanup completed")

# Backward compatibility wrapper
def analyze_transcription_highlights(video_id, input_file=None, output_dir=None):
    """Backward compatibility wrapper for the original function"""
    return analyze_transcription_highlights_safe(video_id, input_file, output_dir)
