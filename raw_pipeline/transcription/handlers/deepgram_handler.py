#!/usr/bin/env python3
"""
Deepgram Handler

Refactored Deepgram transcription handler that integrates with the new
transcription router while maintaining backward compatibility.
"""

import asyncio
import os
from typing import Dict
from pathlib import Path

from utils.logging_setup import setup_logger

logger = setup_logger("deepgram_handler", "deepgram_handler.log")

class DeepgramHandler:
    """Deepgram transcription handler with router integration"""
    
    def __init__(self):
        self.api_key = os.getenv('DEEPGRAM_API_KEY')
        if not self.api_key:
            logger.warning("DEEPGRAM_API_KEY not found in environment")
        
        logger.info("DeepgramHandler initialized")
    
    async def process_audio_files(self, video_id: str, audio_file_path: str, 
                                output_dir: str = None) -> Dict:
        """
        Process audio file using Deepgram API
        
        Args:
            video_id (str): Video identifier
            audio_file_path (str): Path to audio file
            output_dir (str): Output directory for results
            
        Returns:
            Dict: Transcription results with file paths
        """
        
        try:
            # Import the existing transcriber to maintain compatibility
            from raw_pipeline.transcriber import TranscriptionHandler as LegacyTranscriber
            
            # Use existing Deepgram implementation
            legacy_transcriber = LegacyTranscriber()
            
            # Call existing method
            result = await legacy_transcriber.process_audio_files(
                video_id=video_id,
                audio_file_path=audio_file_path,
                output_dir=output_dir
            )
            
            logger.info(f"Deepgram transcription completed for {video_id}")
            return result
            
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "method": "deepgram"
            }
    
    def is_available(self) -> bool:
        """Check if Deepgram is available (API key present)"""
        return self.api_key is not None
    
    def get_estimated_cost(self, duration_seconds: float) -> float:
        """Get estimated cost for Deepgram transcription"""
        duration_minutes = duration_seconds / 60.0
        return duration_minutes * 0.0045  # $0.0045 per minute
    
    def get_estimated_time(self, duration_seconds: float) -> float:
        """Get estimated processing time for Deepgram"""
        return 10.0  # Deepgram is typically ~10 seconds regardless of duration
