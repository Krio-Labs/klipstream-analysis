#!/usr/bin/env python3
"""
Hybrid Transcription Processor

Combines GPU Parakeet and Deepgram transcription for optimal cost/performance
balance on medium-length audio files (2-4 hours).

Strategy:
- Use Parakeet GPU for first portion (cost-effective)
- Use Deepgram for remaining portion if timeout risk
- Merge results seamlessly
"""

import asyncio
import time
import os
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd

from utils.logging_setup import setup_logger
from utils.audio_utils import get_audio_info
from .parakeet_gpu import GPUOptimizedParakeetTranscriber
from .deepgram_handler import DeepgramHandler

logger = setup_logger("hybrid_processor", "hybrid_processor.log")

class HybridProcessor:
    """Hybrid transcription processor combining Parakeet and Deepgram"""
    
    def __init__(self):
        self.parakeet_handler = None  # Lazy load
        self.deepgram_handler = DeepgramHandler()
        self.split_threshold_hours = 2.5  # Switch to Deepgram after 2.5 hours
        
        logger.info("HybridProcessor initialized")
    
    def _lazy_load_parakeet(self):
        """Lazy load Parakeet handler"""
        if self.parakeet_handler is None:
            try:
                self.parakeet_handler = GPUOptimizedParakeetTranscriber()
                logger.info("Parakeet GPU handler loaded for hybrid processing")
            except Exception as e:
                logger.error(f"Failed to load Parakeet for hybrid processing: {e}")
                self.parakeet_handler = False
        return self.parakeet_handler if self.parakeet_handler is not False else None
    
    async def process_audio_files(self, video_id: str, audio_file_path: str, 
                                output_dir: str = None, audio_info: Dict = None) -> Dict:
        """
        Process audio using hybrid approach
        
        Args:
            video_id (str): Video identifier
            audio_file_path (str): Path to audio file
            output_dir (str): Output directory
            audio_info (Dict): Audio file information
            
        Returns:
            Dict: Combined transcription results
        """
        
        start_time = time.time()
        
        try:
            # Get audio info if not provided
            if not audio_info:
                audio_info = get_audio_info(audio_file_path)
            
            duration_hours = audio_info["duration_hours"]
            
            logger.info(f"Starting hybrid processing for {duration_hours:.2f}h audio")
            
            # Determine split strategy
            if duration_hours <= self.split_threshold_hours:
                # Use Parakeet for entire file
                return await self._process_with_parakeet_only(
                    video_id, audio_file_path, output_dir
                )
            else:
                # Use hybrid approach
                return await self._process_with_hybrid_split(
                    video_id, audio_file_path, output_dir, audio_info
                )
                
        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")
            # Fallback to Deepgram for entire file
            logger.info("Falling back to Deepgram for entire file")
            return await self.deepgram_handler.process_audio_files(
                video_id, audio_file_path, output_dir
            )
    
    async def _process_with_parakeet_only(self, video_id: str, audio_file_path: str, 
                                        output_dir: str) -> Dict:
        """Process entire file with Parakeet GPU"""
        
        parakeet = self._lazy_load_parakeet()
        if not parakeet:
            # Fallback to Deepgram
            logger.warning("Parakeet not available, using Deepgram")
            return await self.deepgram_handler.process_audio_files(
                video_id, audio_file_path, output_dir
            )
        
        logger.info("Processing entire file with Parakeet GPU")
        result = await parakeet.transcribe_long_audio(audio_file_path)
        
        # Convert to standard format
        return await self._convert_parakeet_result_to_standard_format(
            result, video_id, output_dir
        )
    
    async def _process_with_hybrid_split(self, video_id: str, audio_file_path: str, 
                                       output_dir: str, audio_info: Dict) -> Dict:
        """Process file using hybrid split approach"""
        
        duration_seconds = audio_info["duration_seconds"]
        split_point_seconds = self.split_threshold_hours * 3600
        
        logger.info(f"Splitting audio at {split_point_seconds/3600:.1f} hours")
        
        # Split audio file
        first_part_path, second_part_path = await self._split_audio_file(
            audio_file_path, split_point_seconds
        )
        
        try:
            # Process first part with Parakeet
            logger.info("Processing first part with Parakeet GPU")
            parakeet_task = self._process_part_with_parakeet(first_part_path)
            
            # Process second part with Deepgram
            logger.info("Processing second part with Deepgram")
            deepgram_task = self._process_part_with_deepgram(second_part_path)
            
            # Run both in parallel
            parakeet_result, deepgram_result = await asyncio.gather(
                parakeet_task, deepgram_task, return_exceptions=True
            )
            
            # Handle any exceptions
            if isinstance(parakeet_result, Exception):
                logger.error(f"Parakeet processing failed: {parakeet_result}")
                # Use Deepgram for entire file
                return await self.deepgram_handler.process_audio_files(
                    video_id, audio_file_path, output_dir
                )
            
            if isinstance(deepgram_result, Exception):
                logger.error(f"Deepgram processing failed: {deepgram_result}")
                # Use only Parakeet result (truncated)
                return await self._convert_parakeet_result_to_standard_format(
                    parakeet_result, video_id, output_dir
                )
            
            # Merge results
            merged_result = await self._merge_transcription_results(
                parakeet_result, deepgram_result, split_point_seconds
            )
            
            # Save merged results
            return await self._save_merged_results(
                merged_result, video_id, output_dir
            )
            
        finally:
            # Clean up temporary files
            self._cleanup_temp_files([first_part_path, second_part_path])
    
    async def _split_audio_file(self, audio_file_path: str, 
                              split_point_seconds: float) -> Tuple[str, str]:
        """Split audio file at specified point"""
        
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(audio_file_path)
            split_point_ms = int(split_point_seconds * 1000)
            
            # Split audio
            first_part = audio[:split_point_ms]
            second_part = audio[split_point_ms:]
            
            # Save parts
            base_path = Path(audio_file_path).stem
            first_part_path = f"/tmp/{base_path}_part1.wav"
            second_part_path = f"/tmp/{base_path}_part2.wav"
            
            first_part.export(first_part_path, format="wav")
            second_part.export(second_part_path, format="wav")
            
            logger.info(f"Audio split: {first_part_path}, {second_part_path}")
            return first_part_path, second_part_path
            
        except Exception as e:
            logger.error(f"Failed to split audio: {e}")
            raise
    
    async def _process_part_with_parakeet(self, audio_path: str) -> Dict:
        """Process audio part with Parakeet"""
        
        parakeet = self._lazy_load_parakeet()
        if not parakeet:
            raise RuntimeError("Parakeet not available")
        
        return await parakeet.transcribe_long_audio(audio_path)
    
    async def _process_part_with_deepgram(self, audio_path: str) -> Dict:
        """Process audio part with Deepgram"""
        
        # Create temporary video ID for this part
        temp_video_id = f"hybrid_part_{int(time.time())}"
        temp_output_dir = f"/tmp/hybrid_output_{temp_video_id}"
        
        result = await self.deepgram_handler.process_audio_files(
            temp_video_id, audio_path, temp_output_dir
        )
        
        # Load and return the actual transcription data
        if "words_file" in result:
            words_df = pd.read_csv(result["words_file"])
            paragraphs_df = pd.read_csv(result["paragraphs_file"])
            
            return {
                "words": words_df.to_dict('records'),
                "segments": paragraphs_df.to_dict('records'),
                "text": " ".join(words_df["word"].tolist())
            }
        
        return {"words": [], "segments": [], "text": ""}
    
    async def _merge_transcription_results(self, parakeet_result: Dict, 
                                         deepgram_result: Dict, 
                                         split_point_seconds: float) -> Dict:
        """Merge Parakeet and Deepgram results"""
        
        # Adjust Deepgram timestamps to account for split
        adjusted_deepgram_words = []
        for word in deepgram_result.get("words", []):
            adjusted_word = word.copy()
            adjusted_word["start"] = word.get("start", 0) + split_point_seconds
            adjusted_word["end"] = word.get("end", 0) + split_point_seconds
            adjusted_deepgram_words.append(adjusted_word)
        
        adjusted_deepgram_segments = []
        for segment in deepgram_result.get("segments", []):
            adjusted_segment = segment.copy()
            adjusted_segment["start"] = segment.get("start", 0) + split_point_seconds
            adjusted_segment["end"] = segment.get("end", 0) + split_point_seconds
            adjusted_deepgram_segments.append(adjusted_segment)
        
        # Combine results
        merged_words = parakeet_result.get("words", []) + adjusted_deepgram_words
        merged_segments = parakeet_result.get("segments", []) + adjusted_deepgram_segments
        merged_text = parakeet_result.get("text", "") + " " + deepgram_result.get("text", "")
        
        return {
            "words": merged_words,
            "segments": merged_segments,
            "text": merged_text.strip()
        }
    
    async def _convert_parakeet_result_to_standard_format(self, parakeet_result: Dict, 
                                                        video_id: str, 
                                                        output_dir: str) -> Dict:
        """Convert Parakeet result to standard format"""
        
        if not output_dir:
            output_dir = f"output/transcripts/{video_id}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save words CSV
        words_file = f"{output_dir}/{video_id}_words.csv"
        words_df = pd.DataFrame(parakeet_result.get("words", []))
        if not words_df.empty:
            words_df.to_csv(words_file, index=False)
        
        # Save paragraphs CSV
        paragraphs_file = f"{output_dir}/{video_id}_paragraphs.csv"
        segments_df = pd.DataFrame(parakeet_result.get("segments", []))
        if not segments_df.empty:
            segments_df.to_csv(paragraphs_file, index=False)
        
        # Save JSON
        json_file = f"{output_dir}/{video_id}_transcript.json"
        transcript_json = {
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": parakeet_result.get("text", ""),
                        "words": parakeet_result.get("words", [])
                    }]
                }]
            }
        }
        
        import json
        with open(json_file, 'w') as f:
            json.dump(transcript_json, f, indent=2)
        
        return {
            "words_file": words_file,
            "paragraphs_file": paragraphs_file,
            "transcript_json_file": json_file,
            "status": "completed",
            "method": "parakeet_gpu"
        }
    
    async def _save_merged_results(self, merged_result: Dict, video_id: str, 
                                 output_dir: str) -> Dict:
        """Save merged transcription results"""
        
        if not output_dir:
            output_dir = f"output/transcripts/{video_id}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save words CSV
        words_file = f"{output_dir}/{video_id}_words.csv"
        words_df = pd.DataFrame(merged_result["words"])
        words_df.to_csv(words_file, index=False)
        
        # Save paragraphs CSV
        paragraphs_file = f"{output_dir}/{video_id}_paragraphs.csv"
        segments_df = pd.DataFrame(merged_result["segments"])
        segments_df.to_csv(paragraphs_file, index=False)
        
        # Save JSON
        json_file = f"{output_dir}/{video_id}_transcript.json"
        transcript_json = {
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": merged_result["text"],
                        "words": merged_result["words"]
                    }]
                }]
            }
        }
        
        import json
        with open(json_file, 'w') as f:
            json.dump(transcript_json, f, indent=2)
        
        logger.info(f"Hybrid transcription completed: {len(merged_result['words'])} words")
        
        return {
            "words_file": words_file,
            "paragraphs_file": paragraphs_file,
            "transcript_json_file": json_file,
            "status": "completed",
            "method": "hybrid"
        }
    
    def _cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")
    
    def get_estimated_cost(self, duration_seconds: float) -> float:
        """Get estimated cost for hybrid processing"""
        
        if duration_seconds <= self.split_threshold_hours * 3600:
            # Parakeet only
            processing_time_hours = duration_seconds / (40 * 3600)  # 40x real-time
            return processing_time_hours * 0.45  # GPU cost
        else:
            # Hybrid: Parakeet for first part, Deepgram for second
            parakeet_duration = self.split_threshold_hours * 3600
            deepgram_duration = duration_seconds - parakeet_duration
            
            parakeet_cost = (parakeet_duration / (40 * 3600)) * 0.45
            deepgram_cost = (deepgram_duration / 60) * 0.0045
            
            return parakeet_cost + deepgram_cost
    
    def get_estimated_time(self, duration_seconds: float) -> float:
        """Get estimated processing time for hybrid approach"""
        
        if duration_seconds <= self.split_threshold_hours * 3600:
            # Parakeet only
            return duration_seconds / 40.0  # 40x real-time
        else:
            # Hybrid: parallel processing
            parakeet_time = (self.split_threshold_hours * 3600) / 40.0
            deepgram_time = 10.0  # Deepgram baseline
            
            # Return max since they run in parallel
            return max(parakeet_time, deepgram_time)
