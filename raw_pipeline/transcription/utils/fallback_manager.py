#!/usr/bin/env python3
"""
Fallback Manager

Handles fallback mechanisms when transcription methods fail, providing
robust error recovery and method switching capabilities.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable
from enum import Enum

from utils.logging_setup import setup_logger

logger = setup_logger("fallback_manager", "fallback_manager.log")

class FailureType(Enum):
    """Types of transcription failures"""
    GPU_MEMORY_ERROR = "gpu_memory_error"
    MODEL_LOADING_ERROR = "model_loading_error"
    NETWORK_ERROR = "network_error"
    PROCESSING_TIMEOUT = "processing_timeout"
    FILE_ERROR = "file_error"
    UNKNOWN_ERROR = "unknown_error"

class FallbackManager:
    """Manages fallback mechanisms for transcription failures"""
    
    def __init__(self):
        self.fallback_chain = [
            "parakeet_gpu",
            "parakeet_cpu", 
            "deepgram",
            "error"
        ]
        
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0
        }
        
        self.failure_counts = {}
        self.last_failure_times = {}
        
        logger.info("FallbackManager initialized")
    
    async def handle_method_failure(self, failed_method: str, error: Exception,
                                  audio_file_path: str, video_id: str, 
                                  output_dir: str) -> Dict:
        """
        Handle failure of a specific transcription method
        
        Args:
            failed_method (str): Method that failed
            error (Exception): The error that occurred
            audio_file_path (str): Path to audio file
            video_id (str): Video identifier
            output_dir (str): Output directory
            
        Returns:
            Dict: Result from fallback method or error
        """
        
        failure_type = self._classify_failure(error)
        logger.warning(f"Method {failed_method} failed with {failure_type.value}: {error}")
        
        # Record failure
        self._record_failure(failed_method, failure_type)
        
        # Determine next method in fallback chain
        next_method = self._get_next_fallback_method(failed_method, failure_type)
        
        if next_method == "error":
            return await self.handle_complete_failure(
                error, audio_file_path, video_id, output_dir
            )
        
        # Attempt fallback with retry logic
        return await self._attempt_fallback_with_retry(
            next_method, audio_file_path, video_id, output_dir, failure_type
        )
    
    async def handle_complete_failure(self, error: Exception, audio_file_path: str,
                                    video_id: str, output_dir: str) -> Dict:
        """Handle complete failure when all methods have failed"""
        
        logger.error(f"Complete transcription failure for {video_id}: {error}")
        
        return {
            "error": f"All transcription methods failed: {str(error)}",
            "status": "failed",
            "video_id": video_id,
            "audio_file": audio_file_path,
            "failure_time": time.time(),
            "fallback_exhausted": True
        }
    
    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify the type of failure based on error"""
        
        error_str = str(error).lower()
        
        if "cuda" in error_str or "memory" in error_str or "out of memory" in error_str:
            return FailureType.GPU_MEMORY_ERROR
        elif "model" in error_str or "loading" in error_str or "download" in error_str:
            return FailureType.MODEL_LOADING_ERROR
        elif "network" in error_str or "connection" in error_str or "timeout" in error_str:
            return FailureType.NETWORK_ERROR
        elif "timeout" in error_str or "time" in error_str:
            return FailureType.PROCESSING_TIMEOUT
        elif "file" in error_str or "path" in error_str or "not found" in error_str:
            return FailureType.FILE_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    def _get_next_fallback_method(self, failed_method: str, 
                                failure_type: FailureType) -> str:
        """Determine next method in fallback chain"""
        
        try:
            current_index = self.fallback_chain.index(failed_method)
        except ValueError:
            # Method not in chain, start from beginning
            current_index = -1
        
        # Special handling for specific failure types
        if failure_type == FailureType.GPU_MEMORY_ERROR:
            # Skip GPU methods
            if failed_method == "parakeet_gpu":
                return "deepgram"  # Skip CPU Parakeet, go directly to Deepgram
        
        elif failure_type == FailureType.MODEL_LOADING_ERROR:
            # Skip all Parakeet methods
            return "deepgram"
        
        elif failure_type == FailureType.NETWORK_ERROR:
            # If Deepgram failed due to network, no fallback available
            if failed_method == "deepgram":
                return "error"
        
        # Default fallback chain progression
        next_index = current_index + 1
        if next_index >= len(self.fallback_chain):
            return "error"
        
        return self.fallback_chain[next_index]
    
    async def _attempt_fallback_with_retry(self, method: str, audio_file_path: str,
                                         video_id: str, output_dir: str,
                                         original_failure: FailureType) -> Dict:
        """Attempt fallback method with retry logic"""
        
        max_retries = self.retry_config["max_retries"]
        base_delay = self.retry_config["base_delay"]
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Attempting fallback to {method} (attempt {attempt + 1})")
                
                # Execute fallback method
                result = await self._execute_fallback_method(
                    method, audio_file_path, video_id, output_dir
                )
                
                if result and "error" not in result:
                    logger.info(f"Fallback to {method} successful")
                    result["fallback_method"] = method
                    result["fallback_attempt"] = attempt + 1
                    return result
                
            except Exception as e:
                logger.warning(f"Fallback attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (self.retry_config["exponential_base"] ** attempt),
                        self.retry_config["max_delay"]
                    )
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed, try next in chain
                    next_method = self._get_next_fallback_method(method, original_failure)
                    if next_method != "error":
                        return await self._attempt_fallback_with_retry(
                            next_method, audio_file_path, video_id, output_dir, original_failure
                        )
        
        # All fallback attempts failed
        return await self.handle_complete_failure(
            Exception(f"Fallback method {method} failed after {max_retries} retries"),
            audio_file_path, video_id, output_dir
        )
    
    async def _execute_fallback_method(self, method: str, audio_file_path: str,
                                     video_id: str, output_dir: str) -> Dict:
        """Execute specific fallback method"""
        
        if method == "parakeet_gpu":
            from ..handlers.parakeet_gpu import GPUOptimizedParakeetTranscriber
            handler = GPUOptimizedParakeetTranscriber()
            result = await handler.transcribe_long_audio(audio_file_path)
            return await self._convert_to_standard_format(result, video_id, output_dir, "parakeet_gpu")
        
        elif method == "parakeet_cpu":
            # Use CPU-only Parakeet (if available)
            logger.warning("CPU Parakeet not implemented, falling back to Deepgram")
            method = "deepgram"
        
        if method == "deepgram":
            from ..handlers.deepgram_handler import DeepgramHandler
            handler = DeepgramHandler()
            return await handler.process_audio_files(video_id, audio_file_path, output_dir)
        
        raise ValueError(f"Unknown fallback method: {method}")
    
    async def _convert_to_standard_format(self, result: Dict, video_id: str,
                                        output_dir: str, method: str) -> Dict:
        """Convert Parakeet result to standard format"""
        
        import os
        import pandas as pd
        import json
        
        if not output_dir:
            output_dir = f"output/transcripts/{video_id}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save words CSV
        words_file = f"{output_dir}/{video_id}_words.csv"
        words_df = pd.DataFrame(result.get("words", []))
        if not words_df.empty:
            words_df.to_csv(words_file, index=False)
        
        # Save paragraphs CSV
        paragraphs_file = f"{output_dir}/{video_id}_paragraphs.csv"
        segments_df = pd.DataFrame(result.get("segments", []))
        if not segments_df.empty:
            segments_df.to_csv(paragraphs_file, index=False)
        
        # Save JSON
        json_file = f"{output_dir}/{video_id}_transcript.json"
        transcript_json = {
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": result.get("text", ""),
                        "words": result.get("words", [])
                    }]
                }]
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(transcript_json, f, indent=2)
        
        return {
            "words_file": words_file,
            "paragraphs_file": paragraphs_file,
            "transcript_json_file": json_file,
            "status": "completed",
            "method": method
        }
    
    def _record_failure(self, method: str, failure_type: FailureType):
        """Record failure for monitoring and analysis"""
        
        current_time = time.time()
        
        # Update failure counts
        key = f"{method}_{failure_type.value}"
        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
        self.last_failure_times[key] = current_time
        
        logger.info(f"Recorded failure: {key} (count: {self.failure_counts[key]})")
    
    def get_failure_statistics(self) -> Dict:
        """Get failure statistics for monitoring"""
        
        return {
            "failure_counts": self.failure_counts.copy(),
            "last_failure_times": self.last_failure_times.copy(),
            "fallback_chain": self.fallback_chain.copy(),
            "retry_config": self.retry_config.copy()
        }
    
    def reset_failure_statistics(self):
        """Reset failure statistics"""
        
        self.failure_counts.clear()
        self.last_failure_times.clear()
        logger.info("Failure statistics reset")
