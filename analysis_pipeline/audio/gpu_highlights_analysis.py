#!/usr/bin/env python3
"""
GPU-Optimized Highlights Analysis Module

This module provides GPU-accelerated highlights analysis using:
- CuPy for GPU-accelerated numpy operations
- cuDF for GPU-accelerated pandas operations  
- CuSignal for GPU-accelerated signal processing
- PyTorch for advanced GPU computations

Fallback to CPU implementations when GPU is not available.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

# GPU acceleration imports with fallbacks
try:
    import cupy as cp
    import cudf
    import cusignal
    HAS_CUPY = True
    print("✅ CuPy available - GPU acceleration enabled")
except ImportError:
    import numpy as cp  # Fallback to numpy
    cudf = None
    cusignal = None
    HAS_CUPY = False
    print("⚠️ CuPy not available - falling back to CPU")

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
    if HAS_TORCH:
        print(f"✅ PyTorch CUDA available - GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ PyTorch CUDA not available")
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not available")

from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("gpu_highlights_analysis", "gpu_highlights_analysis.log")

class GPUHighlightsAnalyzer:
    """GPU-optimized highlights analysis with automatic CPU fallback"""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPU highlights analyzer
        
        Args:
            use_gpu (bool): Whether to attempt GPU acceleration
        """
        self.use_gpu = use_gpu and HAS_CUPY
        self.device = "cuda" if self.use_gpu else "cpu"
        
        if self.use_gpu:
            try:
                # Initialize GPU
                cp.cuda.Device(0).use()
                self.gpu_memory_pool = cp.get_default_memory_pool()
                logger.info(f"🚀 GPU acceleration enabled - Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}, falling back to CPU")
                self.use_gpu = False
                self.device = "cpu"
        else:
            logger.info("🖥️ Using CPU-only processing")
    
    def analyze_highlights_gpu(self, video_id: str, input_file: str, 
                              output_dir: Optional[str] = None, 
                              timeout: int = 90) -> Optional[pd.DataFrame]:
        """
        GPU-accelerated highlights analysis
        
        Args:
            video_id (str): Video ID
            input_file (str): Path to segments CSV file
            output_dir (str, optional): Output directory
            timeout (int): Timeout in seconds
            
        Returns:
            pd.DataFrame: Top highlights or None if failed
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting GPU-accelerated highlights analysis for video {video_id}")
            
            # Step 1: Load data (GPU-accelerated if cuDF available)
            if self.use_gpu and cudf is not None:
                logger.info("📊 Loading data with cuDF (GPU)")
                data = cudf.read_csv(input_file)
                data.columns = data.columns.str.strip()
                # Convert to pandas for compatibility (cuDF operations are much faster for large datasets)
                data = data.to_pandas()
            else:
                logger.info("📊 Loading data with pandas (CPU)")
                data = pd.read_csv(input_file)
                data.columns = data.columns.str.strip()
            
            logger.info(f"📊 Loaded {len(data)} segments")
            
            # Step 2: GPU-accelerated feature processing
            if self.use_gpu:
                highlights = self._process_features_gpu(data, video_id)
            else:
                highlights = self._process_features_cpu(data, video_id)
            
            # Step 3: GPU-accelerated peak detection
            if self.use_gpu and len(highlights) > 0:
                top_highlights = self._find_peaks_gpu(highlights)
            else:
                top_highlights = self._find_peaks_cpu(highlights)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ GPU highlights analysis completed in {elapsed:.2f}s with {len(top_highlights)} highlights")
            
            return top_highlights
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ GPU highlights analysis failed after {elapsed:.2f}s: {e}")
            return None
        finally:
            # Clean up GPU memory
            if self.use_gpu:
                self._cleanup_gpu_memory()
    
    def _process_features_gpu(self, data: pd.DataFrame, video_id: str) -> pd.DataFrame:
        """GPU-accelerated feature processing using CuPy"""
        logger.info("🔥 Processing features on GPU")
        
        # Convert pandas columns to CuPy arrays for GPU processing
        emotion_cols = ['excitement', 'funny', 'happiness', 'anger', 'sadness']
        
        # Check if emotion columns exist
        available_emotions = [col for col in emotion_cols if col in data.columns]
        
        if available_emotions:
            # GPU-accelerated emotion intensity calculation
            emotion_arrays = []
            for col in available_emotions:
                if col in data.columns:
                    emotion_arrays.append(cp.asarray(data[col].values))
                else:
                    emotion_arrays.append(cp.zeros(len(data)))
            
            # Stack arrays and compute max along emotion axis (GPU operation)
            emotion_matrix = cp.stack(emotion_arrays, axis=1)
            emotion_intensity_gpu = cp.max(emotion_matrix, axis=1)
            
            # Convert back to CPU for pandas compatibility
            data['emotion_intensity'] = cp.asnumpy(emotion_intensity_gpu)
        else:
            logger.warning("No emotion columns found, using fallback")
            data['emotion_intensity'] = 0.5
        
        # GPU-accelerated highlight score calculation
        if 'highlight_score' in data.columns:
            highlight_scores = cp.asarray(data['highlight_score'].values)
            emotion_intensity = cp.asarray(data['emotion_intensity'].values)
            
            # Weighted combination (GPU operation)
            weighted_scores = highlight_scores * 0.8 + emotion_intensity * 0.2
            data['weighted_highlight_score'] = cp.asnumpy(weighted_scores)
        else:
            logger.warning("No highlight_score found, using emotion intensity only")
            data['weighted_highlight_score'] = data['emotion_intensity']
        
        # GPU-accelerated normalization
        if self.use_gpu and len(data) > 1000:  # Only use GPU for larger datasets
            scores = cp.asarray(data['weighted_highlight_score'].values)
            
            # Normalize to [0, 1] range (GPU operation)
            min_score = cp.min(scores)
            max_score = cp.max(scores)
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
                data['weighted_highlight_score'] = cp.asnumpy(normalized_scores)
        
        logger.info(f"🔥 GPU feature processing completed for {len(data)} segments")
        return data
    
    def _process_features_cpu(self, data: pd.DataFrame, video_id: str) -> pd.DataFrame:
        """CPU fallback for feature processing"""
        logger.info("🖥️ Processing features on CPU")
        
        # Standard CPU processing
        emotion_cols = ['excitement', 'funny', 'happiness', 'anger', 'sadness']
        available_emotions = [col for col in emotion_cols if col in data.columns]
        
        if available_emotions:
            data['emotion_intensity'] = data[available_emotions].max(axis=1)
        else:
            data['emotion_intensity'] = 0.5
        
        if 'highlight_score' in data.columns:
            data['weighted_highlight_score'] = (
                data['highlight_score'] * 0.8 + 
                data['emotion_intensity'] * 0.2
            )
        else:
            data['weighted_highlight_score'] = data['emotion_intensity']
        
        return data
    
    def _find_peaks_gpu(self, data: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated peak detection using CuSignal"""
        logger.info("🔍 Finding peaks on GPU")
        
        try:
            if cusignal is not None and len(data) > 100:
                # Use CuSignal for peak detection (GPU-accelerated)
                scores = cp.asarray(data['weighted_highlight_score'].values)
                
                # GPU-accelerated peak finding
                peaks_gpu = cusignal.find_peaks(
                    scores,
                    distance=max(1, len(data) // 20),  # Adaptive distance
                    prominence=0.1,
                    height=cp.percentile(scores, 60)  # Top 40% threshold
                )[0]
                
                # Convert back to CPU
                peaks = cp.asnumpy(peaks_gpu)
                
                if len(peaks) > 0:
                    peak_data = data.iloc[peaks].copy()
                    # Sort by score and take top 10
                    top_highlights = peak_data.nlargest(10, 'weighted_highlight_score')
                else:
                    # Fallback: take top 10 by score
                    top_highlights = data.nlargest(10, 'weighted_highlight_score')
            else:
                # Fallback to CPU method
                top_highlights = self._find_peaks_cpu(data)
                
        except Exception as e:
            logger.warning(f"GPU peak detection failed: {e}, falling back to CPU")
            top_highlights = self._find_peaks_cpu(data)
        
        return top_highlights
    
    def _find_peaks_cpu(self, data: pd.DataFrame) -> pd.DataFrame:
        """CPU fallback for peak detection"""
        logger.info("🔍 Finding peaks on CPU")
        
        # Simple CPU-based peak detection
        try:
            from scipy.signal import find_peaks
            
            scores = data['weighted_highlight_score'].values
            peaks, _ = find_peaks(
                scores,
                distance=max(1, len(data) // 20),
                prominence=0.1,
                height=np.percentile(scores, 60)
            )
            
            if len(peaks) > 0:
                peak_data = data.iloc[peaks].copy()
                top_highlights = peak_data.nlargest(10, 'weighted_highlight_score')
            else:
                top_highlights = data.nlargest(10, 'weighted_highlight_score')
                
        except ImportError:
            # Fallback: simple sorting
            logger.info("SciPy not available, using simple sorting")
            top_highlights = data.nlargest(10, 'weighted_highlight_score')
        
        return top_highlights
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if self.use_gpu:
            try:
                # Clear CuPy memory pool
                self.gpu_memory_pool.free_all_blocks()
                cp.cuda.Stream.null.synchronize()
                logger.debug("🧹 GPU memory cleaned up")
            except Exception as e:
                logger.warning(f"GPU memory cleanup failed: {e}")

# Global GPU analyzer instance
gpu_analyzer = GPUHighlightsAnalyzer()

def analyze_highlights_gpu_optimized(video_id: str, input_file: str, 
                                   output_dir: Optional[str] = None, 
                                   timeout: int = 90) -> Optional[pd.DataFrame]:
    """
    GPU-optimized highlights analysis with automatic CPU fallback
    
    This function provides significant speedup for large datasets while
    maintaining compatibility with systems without GPU acceleration.
    
    Args:
        video_id (str): Video ID
        input_file (str): Path to segments CSV file  
        output_dir (str, optional): Output directory
        timeout (int): Timeout in seconds
        
    Returns:
        pd.DataFrame: Top highlights or None if failed
    """
    return gpu_analyzer.analyze_highlights_gpu(video_id, input_file, output_dir, timeout)
