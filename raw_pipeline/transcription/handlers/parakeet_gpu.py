#!/usr/bin/env python3
"""
GPU-Optimized Parakeet Transcriber

This is an optimized version of the Parakeet transcriber specifically designed
for GPU-enabled Cloud Run environments with batch processing capabilities.

Key optimizations:
- GPU batch processing for multiple chunks
- Adaptive chunking based on available resources
- Parallel processing capabilities
- Memory-efficient handling of long audio files
"""

import asyncio
import os
import csv
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydub import AudioSegment

# Import NeMo if available
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

from utils.logging_setup import setup_logger
from utils.config import RAW_AUDIO_DIR, RAW_TRANSCRIPTS_DIR

logger = setup_logger("parakeet_gpu", "parakeet_gpu.log")

class GPUOptimizedParakeetTranscriber:
    """GPU-optimized Parakeet transcriber with batch processing"""
    
    def __init__(self, model_name="nvidia/parakeet-tdt-0.6b-v2", device=None):
        """Initialize the GPU-optimized Parakeet transcriber"""
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.model = None
        self.batch_size = self._determine_batch_size()
        self._setup_model()
    
    def _get_best_device(self):
        """Determine the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device
    
    def _determine_batch_size(self):
        """Determine optimal batch size based on available GPU memory"""
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 20:  # L4 or better
                return 8
            elif gpu_memory_gb >= 12:  # T4 or similar
                return 4
            else:
                return 2
        return 1  # CPU or MPS
    
    def _setup_model(self):
        """Set up the Parakeet model"""
        try:
            if not NEMO_AVAILABLE:
                raise ImportError("NeMo toolkit required. Install with: pip install nemo_toolkit[asr]")
            
            logger.info(f"Loading Parakeet model: {self.model_name}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
            
            if self.device != "cpu" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"Model loaded on {self.device} with batch size {self.batch_size}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_chunk_parameters(self) -> Tuple[int, int]:
        """Get optimal chunk parameters based on available resources"""
        if torch.cuda.is_available():
            # GPU: Use larger chunks for efficiency
            chunk_duration_ms = 10 * 60 * 1000  # 10 minutes
            overlap_ms = 10 * 1000  # 10 seconds
        else:
            # CPU: Use smaller chunks to avoid memory issues
            chunk_duration_ms = 5 * 60 * 1000  # 5 minutes
            overlap_ms = 5 * 1000  # 5 seconds
        
        return chunk_duration_ms, overlap_ms
    
    def _create_audio_chunks(self, audio_file_path: str, duration_seconds: float) -> List[Dict]:
        """Create audio chunks for processing"""
        audio = AudioSegment.from_file(audio_file_path)
        
        # Convert to optimal format
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        chunk_duration_ms, overlap_ms = self._get_chunk_parameters()
        
        chunks = []
        chunk_start = 0
        chunk_index = 0
        
        logger.info(f"Creating chunks: {chunk_duration_ms/60000:.1f}min duration, {overlap_ms/1000:.1f}s overlap")
        
        while chunk_start < len(audio):
            chunk_end = min(chunk_start + chunk_duration_ms, len(audio))
            chunk = audio[chunk_start:chunk_end]
            
            # Save chunk to temporary file
            temp_chunk_path = f"/tmp/chunk_{chunk_index}.wav"
            chunk.export(temp_chunk_path, format="wav")
            
            chunks.append({
                "path": temp_chunk_path,
                "start_time": chunk_start / 1000.0,
                "end_time": chunk_end / 1000.0,
                "index": chunk_index
            })
            
            chunk_start += chunk_duration_ms - overlap_ms
            chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks for processing")
        return chunks
    
    async def _process_chunks_batch(self, chunks: List[Dict]) -> Tuple[List, List, List]:
        """Process chunks in batches for GPU efficiency"""
        all_words = []
        all_segments = []
        full_text_parts = []
        
        if torch.cuda.is_available() and len(chunks) > 1:
            # GPU batch processing
            for batch_start in range(0, len(chunks), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                
                logger.info(f"Processing GPU batch {batch_start//self.batch_size + 1} ({len(batch_chunks)} chunks)")
                
                try:
                    # Prepare batch
                    batch_paths = [chunk["path"] for chunk in batch_chunks]
                    
                    # Batch transcribe
                    batch_outputs = self.model.transcribe(audio=batch_paths, timestamps=True)
                    
                    # Process results
                    for chunk_info, output in zip(batch_chunks, batch_outputs):
                        self._process_chunk_result(output, chunk_info, all_words, all_segments, full_text_parts)
                        
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}, falling back to individual processing")
                    # Fallback to individual processing
                    for chunk_info in batch_chunks:
                        await self._process_single_chunk(chunk_info, all_words, all_segments, full_text_parts)
                
                finally:
                    # Cleanup batch files
                    for chunk_info in batch_chunks:
                        try:
                            os.remove(chunk_info["path"])
                        except:
                            pass
        else:
            # Sequential processing for CPU
            for chunk_info in chunks:
                await self._process_single_chunk(chunk_info, all_words, all_segments, full_text_parts)
                try:
                    os.remove(chunk_info["path"])
                except:
                    pass
        
        return all_words, all_segments, full_text_parts
    
    def _process_chunk_result(self, output, chunk_info: Dict, all_words: List, all_segments: List, full_text_parts: List):
        """Process result from a single chunk"""
        if output and hasattr(output, 'text'):
            chunk_text = output.text
            if chunk_text:
                full_text_parts.append(chunk_text)
                
                if hasattr(output, 'timestamp') and output.timestamp:
                    # Word timestamps
                    word_timestamps = output.timestamp.get('word', [])
                    for word_info in word_timestamps:
                        all_words.append({
                            "word": word_info.get('word', ''),
                            "start": word_info.get('start', 0.0) + chunk_info['start_time'],
                            "end": word_info.get('end', 0.0) + chunk_info['start_time']
                        })
                    
                    # Segment timestamps
                    segment_timestamps = output.timestamp.get('segment', [])
                    for segment_info in segment_timestamps:
                        all_segments.append({
                            "text": segment_info.get('segment', ''),
                            "start": segment_info.get('start', 0.0) + chunk_info['start_time'],
                            "end": segment_info.get('end', 0.0) + chunk_info['start_time']
                        })
    
    async def _process_single_chunk(self, chunk_info: Dict, all_words: List, all_segments: List, full_text_parts: List):
        """Process a single chunk individually"""
        try:
            output = self.model.transcribe(audio=[chunk_info["path"]], timestamps=True)
            if output and len(output) > 0:
                self._process_chunk_result(output[0], chunk_info, all_words, all_segments, full_text_parts)
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
    
    async def transcribe_long_audio(self, audio_file_path: str) -> Dict:
        """Transcribe long audio file with GPU optimization"""
        try:
            # Get audio info
            audio_info = AudioSegment.from_file(audio_file_path)
            duration_seconds = len(audio_info) / 1000.0
            
            logger.info(f"Transcribing {duration_seconds/3600:.2f}h audio with GPU optimization")
            
            # Create chunks
            chunks = self._create_audio_chunks(audio_file_path, duration_seconds)
            
            # Process chunks
            all_words, all_segments, full_text_parts = await self._process_chunks_batch(chunks)
            
            # Combine results
            full_text = " ".join(full_text_parts)
            
            logger.info(f"GPU transcription completed: {len(all_words)} words, {len(all_segments)} segments")
            
            return {
                "text": full_text,
                "words": all_words,
                "segments": all_segments
            }
            
        except Exception as e:
            logger.error(f"GPU transcription failed: {e}")
            return {"text": "", "words": [], "segments": []}

# Performance test function
async def test_gpu_performance():
    """Test GPU performance with sample audio"""
    transcriber = GPUOptimizedParakeetTranscriber()
    
    # Test with sample file
    test_file = "/Users/aman/Downloads/Matt Rife Only Fans Full Special.mp3"
    if os.path.exists(test_file):
        import time
        start_time = time.time()
        
        result = await transcriber.transcribe_long_audio(test_file)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result["text"]:
            print(f"✅ GPU Test Results:")
            print(f"Processing time: {processing_time:.1f} seconds")
            print(f"Words transcribed: {len(result['words'])}")
            print(f"Text preview: {result['text'][:100]}...")
        else:
            print("❌ GPU test failed")
    else:
        print("Test file not found")

# Alias for compatibility
ParakeetGPUHandler = GPUOptimizedParakeetTranscriber

if __name__ == "__main__":
    asyncio.run(test_gpu_performance())
