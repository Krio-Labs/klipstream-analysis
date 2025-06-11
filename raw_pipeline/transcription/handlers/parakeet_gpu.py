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

    # Reduce NeMo logging verbosity and disable progress bars
    import logging
    nemo_logger = logging.getLogger('nemo_logger')
    nemo_logger.setLevel(logging.WARNING)

    # Disable tqdm progress bars globally
    import os
    os.environ['TQDM_DISABLE'] = '1'

    # Monkey patch tqdm to disable it
    try:
        import tqdm
        original_tqdm = tqdm.tqdm

        class NoOpTqdm:
            def __init__(self, *args, **kwargs):
                self.iterable = args[0] if args else None

            def __iter__(self):
                if self.iterable:
                    return iter(self.iterable)
                return iter([])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def update(self, *args):
                pass

            def close(self):
                pass

        # Replace tqdm with no-op version
        tqdm.tqdm = NoOpTqdm
        tqdm.tqdm.tqdm = NoOpTqdm

    except ImportError:
        pass

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

    def cleanup_gpu_resources(self):
        """Clean up GPU resources and memory after transcription"""
        try:
            print("🧹 Cleaning up GPU resources...", flush=True)

            # Clear the model from memory
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                print("✅ Model cleared from memory", flush=True)

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("✅ CUDA cache cleared", flush=True)

                # Get memory info after cleanup
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"📊 GPU Memory after cleanup: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved", flush=True)

            # Force garbage collection
            import gc
            gc.collect()
            print("✅ Garbage collection completed", flush=True)

        except Exception as e:
            print(f"⚠️  Error during GPU cleanup: {e}", flush=True)
            logger.warning(f"Error during GPU cleanup: {e}")
    
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

        # Simple progress tracking
        total_chunks = len(chunks)
        processed_chunks = 0

        print(f"🎤 Transcribing {total_chunks} chunks...")

        # Print progress every 10% or every chunk if less than 10 chunks
        progress_interval = max(1, total_chunks // 10)

        if torch.cuda.is_available() and len(chunks) > 1:
            # GPU batch processing
            for batch_start in range(0, len(chunks), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]

                try:
                    # Prepare batch
                    batch_paths = [chunk["path"] for chunk in batch_chunks]

                    # Batch transcribe with suppressed output
                    import os
                    from contextlib import redirect_stdout, redirect_stderr

                    with open(os.devnull, 'w') as devnull:
                        with redirect_stdout(devnull), redirect_stderr(devnull):
                            batch_outputs = self.model.transcribe(audio=batch_paths, timestamps=True)

                    # Process results
                    for chunk_info, output in zip(batch_chunks, batch_outputs):
                        self._process_chunk_result(output, chunk_info, all_words, all_segments, full_text_parts)
                        processed_chunks += 1

                        # Print progress
                        if processed_chunks % progress_interval == 0 or processed_chunks == total_chunks:
                            percentage = (processed_chunks / total_chunks) * 100
                            print(f"🎤 Progress: {processed_chunks}/{total_chunks} chunks ({percentage:.0f}%)")

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}, falling back to individual processing")
                    # Fallback to individual processing
                    for chunk_info in batch_chunks:
                        await self._process_single_chunk(chunk_info, all_words, all_segments, full_text_parts)
                        processed_chunks += 1

                        # Print progress
                        if processed_chunks % progress_interval == 0 or processed_chunks == total_chunks:
                            percentage = (processed_chunks / total_chunks) * 100
                            print(f"🎤 Progress: {processed_chunks}/{total_chunks} chunks ({percentage:.0f}%)")

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
                processed_chunks += 1

                # Print progress
                if processed_chunks % progress_interval == 0 or processed_chunks == total_chunks:
                    percentage = (processed_chunks / total_chunks) * 100
                    print(f"🎤 Progress: {processed_chunks}/{total_chunks} chunks ({percentage:.0f}%)")

                try:
                    os.remove(chunk_info["path"])
                except:
                    pass

        # Final progress message
        print(f"✅ Transcription completed: {processed_chunks} chunks processed")

        # CRITICAL: Clean up GPU memory after transcription
        self.cleanup_gpu_resources()

        return all_words, all_segments, full_text_parts
    
    def _process_chunk_result(self, output, chunk_info: Dict, all_words: List, all_segments: List, full_text_parts: List):
        """Process result from a single chunk"""
        try:
            if output and hasattr(output, 'text'):
                chunk_text = output.text

                if chunk_text:
                    full_text_parts.append(chunk_text)

                # NeMo uses 'timestamp' attribute (corrected from debug analysis)
                if hasattr(output, 'timestamp') and output.timestamp:
                    timestamp = output.timestamp
                    # Word timestamps
                    if 'word' in timestamp and timestamp['word']:
                        word_timestamps = timestamp['word']
                        for word_info in word_timestamps:
                            # NeMo word format: {'word': str, 'start': float, 'end': float, 'start_offset': int, 'end_offset': int}
                            # Use 'start' and 'end' for time, not 'start_offset'/'end_offset' (which are character positions)
                            all_words.append({
                                "word": word_info.get('word', ''),
                                "start": word_info.get('start', 0.0) + chunk_info['start_time'],
                                "end": word_info.get('end', 0.0) + chunk_info['start_time']
                            })

                    # Segment timestamps
                    if 'segment' in timestamp and timestamp['segment']:
                        segment_timestamps = timestamp['segment']
                        for segment_info in segment_timestamps:
                            # NeMo segment format: {'segment': str, 'start': float, 'end': float, 'start_offset': int, 'end_offset': int}
                            # Use 'start' and 'end' for time, not 'start_offset'/'end_offset' (which are character positions)
                            all_segments.append({
                                "text": segment_info.get('segment', ''),  # NeMo uses 'segment' key, not 'text'
                                "start": segment_info.get('start', 0.0) + chunk_info['start_time'],
                                "end": segment_info.get('end', 0.0) + chunk_info['start_time']
                            })

                # If no timestamps available, create basic word-level timestamps
                if not all_words and chunk_text:
                    words = chunk_text.split()
                    word_duration = (chunk_info['end_time'] - chunk_info['start_time']) / len(words) if words else 0

                    for i, word in enumerate(words):
                        start_time = chunk_info['start_time'] + (i * word_duration)
                        end_time = start_time + word_duration
                        all_words.append({
                            "word": word,
                            "start": start_time,
                            "end": end_time
                        })

                # If no segments available, create one from the full text
                if not all_segments and chunk_text:
                    all_segments.append({
                        "text": chunk_text,
                        "start": chunk_info['start_time'],
                        "end": chunk_info['end_time']
                    })
            else:
                logger.warning(f"No text found in output or output is None")

        except Exception as e:
            logger.error(f"Error processing chunk result: {e}")
            import traceback
            traceback.print_exc()
    
    async def _process_single_chunk(self, chunk_info: Dict, all_words: List, all_segments: List, full_text_parts: List):
        """Process a single chunk individually"""
        try:
            # Suppress NeMo's output by redirecting stdout/stderr
            import sys
            import os
            from contextlib import redirect_stdout, redirect_stderr

            # Redirect to null device
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
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

            logger.info(f"✅ Transcription completed: {len(all_words)} words, {len(all_segments)} segments")
            
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

class ParakeetGPUHandler:
    """Handler wrapper for the GPU-optimized Parakeet transcriber"""

    def __init__(self):
        self.transcriber = GPUOptimizedParakeetTranscriber()
        logger.info("ParakeetGPUHandler initialized")

    def cleanup_gpu_resources(self):
        """Clean up GPU resources and memory after transcription"""
        if hasattr(self, 'transcriber') and self.transcriber:
            self.transcriber.cleanup_gpu_resources()

    def _create_paragraphs_from_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Group sentence-level segments into meaningful paragraphs

        Args:
            segments: List of sentence-level segments from NeMo

        Returns:
            List of paragraph-level segments
        """
        if not segments:
            return []

        paragraphs = []
        current_paragraph = {
            "text": "",
            "start": None,
            "end": None,
            "sentences": []
        }

        # Parameters for paragraph grouping (pause-based detection)
        pause_threshold = 0.8  # 0.8+ second pause indicates new paragraph
        max_paragraph_duration = 20.0  # Maximum 20 seconds per paragraph (backup limit)
        max_sentences_per_paragraph = 6  # Maximum 6 sentences per paragraph (backup limit)

        for i, segment in enumerate(segments):
            segment_text = segment.get("text", "").strip()
            segment_start = segment.get("start", 0.0)
            segment_end = segment.get("end", 0.0)

            if not segment_text:
                continue

            # Initialize first paragraph
            if current_paragraph["start"] is None:
                current_paragraph["start"] = segment_start
                current_paragraph["text"] = segment_text
                current_paragraph["end"] = segment_end
                current_paragraph["sentences"].append(segment)
                continue

            # Check if we should start a new paragraph
            should_start_new_paragraph = False

            # Check pause duration (gap between segments) - PRIMARY CRITERIA
            pause_duration = segment_start - current_paragraph["end"]
            if pause_duration >= pause_threshold:
                should_start_new_paragraph = True
                logger.debug(f"New paragraph due to pause: {pause_duration:.1f}s (threshold: {pause_threshold}s)")

            # Secondary criteria (only if pause threshold not met)
            if not should_start_new_paragraph:
                # Check paragraph duration (backup limit)
                current_duration = segment_end - current_paragraph["start"]
                if current_duration >= max_paragraph_duration:
                    should_start_new_paragraph = True
                    logger.debug(f"New paragraph due to max duration: {current_duration:.1f}s")

                # Check sentence count (backup limit)
                elif len(current_paragraph["sentences"]) >= max_sentences_per_paragraph:
                    should_start_new_paragraph = True
                    logger.debug(f"New paragraph due to max sentences: {len(current_paragraph['sentences'])}")

                # Check for strong topic changes (only as last resort)
                elif self._detect_strong_topic_change(current_paragraph["text"], segment_text):
                    should_start_new_paragraph = True
                    logger.debug("New paragraph due to strong topic change")

            if should_start_new_paragraph:
                # Finalize current paragraph
                if current_paragraph["text"]:
                    paragraphs.append({
                        "text": current_paragraph["text"],
                        "start": current_paragraph["start"],
                        "end": current_paragraph["end"]
                    })

                # Start new paragraph
                current_paragraph = {
                    "text": segment_text,
                    "start": segment_start,
                    "end": segment_end,
                    "sentences": [segment]
                }
            else:
                # Add to current paragraph
                current_paragraph["text"] += " " + segment_text
                current_paragraph["end"] = segment_end
                current_paragraph["sentences"].append(segment)

        # Add the last paragraph
        if current_paragraph["text"]:
            paragraphs.append({
                "text": current_paragraph["text"],
                "start": current_paragraph["start"],
                "end": current_paragraph["end"]
            })

        logger.info(f"Created {len(paragraphs)} paragraphs from {len(segments)} segments using pause threshold: {pause_threshold}s")

        # Log paragraph statistics
        if paragraphs:
            avg_duration = sum(p["end"] - p["start"] for p in paragraphs) / len(paragraphs)
            logger.info(f"Average paragraph duration: {avg_duration:.1f}s")

        return paragraphs

    def _detect_strong_topic_change(self, current_text: str, new_text: str) -> bool:
        """
        Detect strong topic changes between segments (conservative approach)
        Only triggers for very clear topic transitions

        Args:
            current_text: Current paragraph text
            new_text: New segment text

        Returns:
            bool: True if strong topic change detected
        """
        # Strong topic change indicators (more conservative)
        strong_topic_change_phrases = [
            "now let's", "moving on", "speaking of", "by the way",
            "anyway", "meanwhile", "however", "on the other hand",
            "in other news", "next up", "switching gears"
        ]

        new_text_lower = new_text.lower().strip()

        # Check if new segment starts with strong topic change indicators
        for phrase in strong_topic_change_phrases:
            if new_text_lower.startswith(phrase + " ") or new_text_lower.startswith(phrase + ","):
                return True

        # Check for very clear context switches (question -> unrelated statement)
        current_ends_with_question = current_text.strip().endswith("?")
        if current_ends_with_question and new_text_lower.startswith(("alright", "okay", "well", "so")):
            return True

        return False

    async def process_audio_files(self, video_id: str, audio_file_path: str, output_dir: str = None) -> Dict:
        """
        Process audio files using GPU-optimized Parakeet transcription

        Args:
            video_id (str): Video identifier
            audio_file_path (str): Path to audio file
            output_dir (str): Output directory for results

        Returns:
            Dict: Transcription results with file paths
        """
        try:
            # Determine output directory
            if output_dir:
                output_dir = Path(output_dir)
            else:
                output_dir = RAW_TRANSCRIPTS_DIR

            # Create output directory if it doesn't exist
            output_dir.mkdir(exist_ok=True, parents=True)

            # Get base name for output files
            base_name = f"audio_{video_id}"

            logger.info(f"Starting Parakeet GPU transcription for video {video_id}")

            # Transcribe using GPU optimization
            result = await self.transcriber.transcribe_long_audio(audio_file_path)

            if not result or not result.get("text"):
                raise RuntimeError("Parakeet transcription returned empty result")

            # Save word-level timestamps
            words_file = output_dir / f"{base_name}_words.csv"
            with open(words_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['start_time', 'end_time', 'word'])

                for word in result.get("words", []):
                    writer.writerow([
                        word.get("start", 0.0),
                        word.get("end", 0.0),
                        word.get("word", "")
                    ])

            logger.info(f"Word timestamps saved to: {words_file}")

            # Create meaningful paragraphs from sentence-level segments
            paragraphs_file = output_dir / f"{base_name}_paragraphs.csv"
            with open(paragraphs_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['start_time', 'end_time', 'text'])

                segments = result.get("segments", [])
                if segments:
                    # Group sentence-level segments into meaningful paragraphs
                    paragraphs = self._create_paragraphs_from_segments(segments)

                    for paragraph in paragraphs:
                        writer.writerow([
                            paragraph.get("start", 0.0),
                            paragraph.get("end", 0.0),
                            paragraph.get("text", "")
                        ])

                    logger.info(f"Created {len(paragraphs)} paragraphs from {len(segments)} segments")
                else:
                    # If no segments, create one from full text
                    writer.writerow([0.0, len(result.get("words", [])) * 0.5, result.get("text", "")])
                    logger.info("Created single paragraph from full text")

            logger.info(f"Paragraphs saved to: {paragraphs_file}")

            # Save transcript JSON
            transcript_json_path = output_dir / f"{base_name}_transcript.json"
            with open(transcript_json_path, 'w') as f:
                json.dump({
                    "text": result.get("text", ""),
                    "words": result.get("words", []),
                    "segments": result.get("segments", []),
                    "metadata": {
                        "model": self.transcriber.model_name,
                        "device": self.transcriber.device,
                        "batch_size": self.transcriber.batch_size
                    }
                }, f, indent=2)

            logger.info(f"Transcript JSON saved to: {transcript_json_path}")

            # Calculate transcription metadata
            transcription_metadata = {
                "method_used": "parakeet",
                "model_name": self.transcriber.model_name,
                "device_used": self.transcriber.device,
                "gpu_used": self.transcriber.device in ["cuda", "mps"],
                "batch_size": self.transcriber.batch_size,
                "words_count": len(result.get("words", [])),
                "segments_count": len(result.get("segments", []))
            }

            return {
                "paragraphs_file": paragraphs_file,
                "words_file": words_file,
                "transcript_json_file": transcript_json_path,
                "transcription_metadata": transcription_metadata
            }

        except Exception as e:
            logger.error(f"Parakeet GPU transcription failed: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(test_gpu_performance())
