#!/usr/bin/env python3
"""
Enhanced GPU-Optimized Parakeet Transcriber

This is a comprehensive GPU optimization implementation for the Parakeet transcriber
with advanced features including AMP, memory optimization, parallel chunking, and
device-specific optimizations.

Key Features:
- Automatic Mixed Precision (AMP) support
- Adaptive memory management with fragmentation handling
- Parallel audio chunking with ThreadPoolExecutor
- Device-specific optimizations (CUDA, MPS, CPU)
- Comprehensive performance monitoring and diagnostics
- Robust error handling and fallback mechanisms
"""

import asyncio
import os
import csv
import json
import torch
import numpy as np
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import multiprocessing

# Import NeMo if available
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
    
    # Monkey patch tqdm to disable it
    try:
        import tqdm
        class NoOpTqdm:
            def __init__(self, *args, **kwargs):
                self.iterable = args[0] if args else None
            def __iter__(self):
                return iter(self.iterable) if self.iterable else iter([])
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, *args):
                pass
            def close(self):
                pass
        tqdm.tqdm = NoOpTqdm
        tqdm.tqdm.tqdm = NoOpTqdm
    except ImportError:
        pass
        
except ImportError:
    NEMO_AVAILABLE = False

from utils.logging_setup import setup_logger
from utils.config import RAW_AUDIO_DIR, RAW_TRANSCRIPTS_DIR

logger = setup_logger("enhanced_parakeet_gpu", "enhanced_parakeet_gpu.log")

class GPUOptimizationConfig:
    """Configuration class for GPU optimization settings"""
    
    def __init__(self):
        # AMP Configuration
        self.enable_amp = self._get_bool_env("ENABLE_AMP", True)
        self.amp_dtype = torch.float16
        
        # Memory Optimization
        self.enable_memory_optimization = self._get_bool_env("ENABLE_MEMORY_OPTIMIZATION", True)
        self.memory_cleanup_threshold = 0.8  # Clean when 80% full
        self.max_memory_fragmentation = 0.3  # 30% fragmentation limit
        
        # Parallel Processing
        self.enable_parallel_chunking = self._get_bool_env("ENABLE_PARALLEL_CHUNKING", True)
        self.max_chunk_workers = min(multiprocessing.cpu_count(), 8)
        self.chunk_io_throttle_delay = 0.01  # 10ms delay between I/O operations
        
        # Device Optimization
        self.enable_device_optimization = self._get_bool_env("ENABLE_DEVICE_OPTIMIZATION", True)
        self.cuda_benchmark = True
        self.cuda_tf32 = True  # For Ampere+ GPUs
        
        # Performance Monitoring
        self.enable_performance_monitoring = self._get_bool_env("ENABLE_PERFORMANCE_MONITORING", True)
        self.memory_monitoring_interval = 1.0  # seconds
        
        # Debugging
        self.enable_debug_logging = self._get_bool_env("ENABLE_DEBUG_LOGGING", False)
        self.save_performance_metrics = self._get_bool_env("SAVE_PERFORMANCE_METRICS", True)
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

class PerformanceMonitor:
    """Performance monitoring and diagnostics"""
    
    def __init__(self, config: GPUOptimizationConfig):
        self.config = config
        self.metrics = {
            "memory_usage": [],
            "processing_times": [],
            "batch_sizes": [],
            "amp_usage": [],
            "device_utilization": []
        }
        self.start_time = None
        self.peak_memory = 0
        self._monitoring_active = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.config.enable_performance_monitoring:
            return
            
        self.start_time = time.time()
        self._monitoring_active = True
        
        if torch.cuda.is_available():
            self._monitor_thread = threading.Thread(target=self._monitor_gpu_memory, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_gpu_memory(self):
        """Monitor GPU memory usage in background thread"""
        while self._monitoring_active:
            try:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    self.peak_memory = max(self.peak_memory, allocated)
                    
                    self.metrics["memory_usage"].append({
                        "timestamp": time.time() - self.start_time,
                        "allocated_gb": allocated,
                        "reserved_gb": reserved
                    })
                
                time.sleep(self.config.memory_monitoring_interval)
            except Exception as e:
                if self.config.enable_debug_logging:
                    logger.debug(f"Memory monitoring error: {e}")
                break
    
    def record_batch_processing(self, batch_size: int, processing_time: float, 
                              amp_enabled: bool, device: str):
        """Record batch processing metrics"""
        self.metrics["batch_sizes"].append(batch_size)
        self.metrics["processing_times"].append(processing_time)
        self.metrics["amp_usage"].append(amp_enabled)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Batch processed: size={batch_size}, time={processing_time:.2f}s, "
                        f"AMP={amp_enabled}, device={device}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            "total_processing_time": total_time,
            "peak_memory_gb": self.peak_memory,
            "average_batch_size": np.mean(self.metrics["batch_sizes"]) if self.metrics["batch_sizes"] else 0,
            "total_batches": len(self.metrics["batch_sizes"]),
            "amp_usage_percentage": (sum(self.metrics["amp_usage"]) / len(self.metrics["amp_usage"]) * 100) 
                                   if self.metrics["amp_usage"] else 0,
            "memory_efficiency": self._calculate_memory_efficiency()
        }
        
        return summary
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        if not self.metrics["memory_usage"]:
            return 0.0
        
        # Calculate average memory utilization vs peak
        avg_usage = np.mean([m["allocated_gb"] for m in self.metrics["memory_usage"]])
        return (avg_usage / self.peak_memory * 100) if self.peak_memory > 0 else 0.0

class EnhancedGPUOptimizedParakeetTranscriber:
    """Enhanced GPU-optimized Parakeet transcriber with comprehensive optimizations"""
    
    def __init__(self, model_name="nvidia/parakeet-tdt-0.6b-v2", device=None, config=None):
        """Initialize the enhanced GPU-optimized Parakeet transcriber"""
        self.model_name = model_name
        self.config = config or GPUOptimizationConfig()
        self.device = device or self._get_best_device()
        self.model = None
        self.batch_size = self._determine_optimal_batch_size()
        self.performance_monitor = PerformanceMonitor(self.config)
        
        # AMP scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self._supports_amp() else None
        
        # Thread pool for parallel chunking
        self.chunk_executor = None
        
        self._setup_device_optimizations()
        self._setup_model()
        
        logger.info(f"Enhanced GPU transcriber initialized: device={self.device}, "
                   f"batch_size={self.batch_size}, AMP={'enabled' if self.scaler else 'disabled'}")
    
    def _get_best_device(self) -> str:
        """Determine the best available device with enhanced detection"""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            compute_capability = torch.cuda.get_device_capability()
            
            logger.info(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB, "
                       f"Compute {compute_capability[0]}.{compute_capability[1]})")
            
            # Check for Ampere+ architecture for TF32 support
            if compute_capability[0] >= 8:
                logger.info("Ampere+ GPU detected - TF32 optimizations available")
                
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            cpu_count = multiprocessing.cpu_count()
            logger.info(f"Using CPU for inference ({cpu_count} cores)")
        
        return device

    def _supports_amp(self) -> bool:
        """Check if device supports Automatic Mixed Precision"""
        if not self.config.enable_amp:
            return False

        if self.device == "cuda":
            # Check for Tensor Core support (compute capability 7.0+)
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability()
                return compute_capability[0] >= 7

        return False

    def _determine_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on device and memory"""
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Enhanced batch size calculation based on GPU memory and architecture
            if gpu_memory_gb >= 40:  # A100, H100
                base_batch_size = 16
            elif gpu_memory_gb >= 24:  # L4, RTX 4090
                base_batch_size = 12
            elif gpu_memory_gb >= 16:  # RTX 4080, V100
                base_batch_size = 8
            elif gpu_memory_gb >= 12:  # T4, RTX 3080
                base_batch_size = 6
            elif gpu_memory_gb >= 8:   # RTX 3070, GTX 1080 Ti
                base_batch_size = 4
            else:
                base_batch_size = 2

            # Reduce batch size if AMP is disabled (uses more memory)
            if not self._supports_amp():
                base_batch_size = max(1, base_batch_size // 2)

            return base_batch_size

        elif self.device == "mps":
            # Apple Silicon - conservative batch sizes
            return 4
        else:
            # CPU - single batch processing
            return 1

    def _setup_device_optimizations(self):
        """Setup device-specific optimizations"""
        if not self.config.enable_device_optimization:
            return

        if self.device == "cuda" and torch.cuda.is_available():
            # Enable cuDNN benchmark for consistent input sizes
            if self.config.cuda_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmark enabled")

            # Enable TF32 for Ampere+ GPUs
            if self.config.cuda_tf32:
                compute_capability = torch.cuda.get_device_capability()
                if compute_capability[0] >= 8:  # Ampere+
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("TF32 optimizations enabled for Ampere+ GPU")

            # Set memory allocation strategy
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

        elif self.device == "mps":
            # Apple Silicon optimizations
            logger.info("MPS optimizations applied")

        elif self.device == "cpu":
            # CPU optimizations
            cpu_cores = multiprocessing.cpu_count()
            torch.set_num_threads(min(cpu_cores, 8))  # Limit to prevent oversubscription
            torch.set_num_interop_threads(min(cpu_cores // 2, 4))
            logger.info(f"CPU optimizations: {torch.get_num_threads()} threads")

    def _setup_model(self):
        """Set up the Parakeet model with optimizations"""
        try:
            if not NEMO_AVAILABLE:
                raise ImportError("NeMo toolkit required. Install with: pip install nemo_toolkit[asr]")

            logger.info(f"Loading Parakeet model: {self.model_name}")

            # Load model
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)

            # Move to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

            # Enable optimizations
            if self.device == "cuda":
                # Enable memory efficient attention if available
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.debug(f"torch.compile not available: {e}")

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @contextmanager
    def _amp_context(self):
        """Context manager for Automatic Mixed Precision"""
        if self.scaler and self.device == "cuda":
            with torch.amp.autocast('cuda', dtype=self.config.amp_dtype):
                yield True
        else:
            yield False

    def _cleanup_memory(self, force: bool = False):
        """Enhanced memory cleanup with fragmentation handling"""
        if not self.config.enable_memory_optimization and not force:
            return

        try:
            if torch.cuda.is_available():
                # Get memory stats before cleanup
                allocated_before = torch.cuda.memory_allocated() / (1024**3)
                reserved_before = torch.cuda.memory_reserved() / (1024**3)

                # Check if cleanup is needed
                memory_usage = allocated_before / (torch.cuda.get_device_properties(0).total_memory / (1024**3))

                if memory_usage > self.config.memory_cleanup_threshold or force:
                    # Clear cache
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Get memory stats after cleanup
                    allocated_after = torch.cuda.memory_allocated() / (1024**3)
                    reserved_after = torch.cuda.memory_reserved() / (1024**3)

                    freed_memory = reserved_before - reserved_after

                    if self.config.enable_debug_logging:
                        logger.debug(f"Memory cleanup: freed {freed_memory:.2f}GB "
                                   f"({allocated_before:.2f}→{allocated_after:.2f}GB allocated)")

                    # Check for memory fragmentation
                    fragmentation = (reserved_after - allocated_after) / reserved_after if reserved_after > 0 else 0
                    if fragmentation > self.config.max_memory_fragmentation:
                        logger.warning(f"High memory fragmentation detected: {fragmentation:.1%}")

        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

    def _get_chunk_parameters(self) -> Tuple[int, int]:
        """Get optimal chunk parameters based on device and memory"""
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Larger chunks for high-memory GPUs
            if gpu_memory_gb >= 24:
                chunk_duration_ms = 15 * 60 * 1000  # 15 minutes
                overlap_ms = 15 * 1000  # 15 seconds
            elif gpu_memory_gb >= 16:
                chunk_duration_ms = 12 * 60 * 1000  # 12 minutes
                overlap_ms = 12 * 1000  # 12 seconds
            else:
                chunk_duration_ms = 8 * 60 * 1000   # 8 minutes
                overlap_ms = 8 * 1000   # 8 seconds

        elif self.device == "mps":
            # Apple Silicon - moderate chunks
            chunk_duration_ms = 8 * 60 * 1000   # 8 minutes
            overlap_ms = 8 * 1000   # 8 seconds
        else:
            # CPU - smaller chunks
            chunk_duration_ms = 5 * 60 * 1000   # 5 minutes
            overlap_ms = 5 * 1000   # 5 seconds

        return chunk_duration_ms, overlap_ms

    def _create_audio_chunks_parallel(self, audio_file_path: str, duration_seconds: float) -> List[Dict]:
        """Create audio chunks using parallel processing"""
        audio = AudioSegment.from_file(audio_file_path)
        chunk_duration_ms, overlap_ms = self._get_chunk_parameters()

        chunks = []
        chunk_start = 0
        chunk_index = 0

        logger.info(f"Creating chunks: {chunk_duration_ms/60000:.1f}min duration, "
                   f"{overlap_ms/1000:.1f}s overlap")

        # Calculate all chunk parameters first
        chunk_params = []
        while chunk_start < len(audio):
            chunk_end = min(chunk_start + chunk_duration_ms, len(audio))
            chunk_params.append({
                "start": chunk_start,
                "end": chunk_end,
                "index": chunk_index,
                "start_time": chunk_start / 1000.0,
                "end_time": chunk_end / 1000.0
            })
            chunk_start += chunk_duration_ms - overlap_ms
            chunk_index += 1

        if self.config.enable_parallel_chunking and len(chunk_params) > 1:
            # Parallel chunk creation
            chunks = self._create_chunks_parallel_worker(audio, chunk_params)
        else:
            # Sequential chunk creation
            chunks = self._create_chunks_sequential(audio, chunk_params)

        logger.info(f"Created {len(chunks)} chunks for processing")
        return chunks

    def _create_chunks_parallel_worker(self, audio: AudioSegment, chunk_params: List[Dict]) -> List[Dict]:
        """Create chunks using parallel workers"""
        chunks = []
        max_workers = min(self.config.max_chunk_workers, len(chunk_params))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit chunk creation tasks
            future_to_params = {
                executor.submit(self._create_single_chunk, audio, params): params
                for params in chunk_params
            }

            # Collect results with progress tracking
            completed = 0
            total = len(chunk_params)

            for future in as_completed(future_to_params):
                try:
                    chunk_info = future.result()
                    if chunk_info:
                        chunks.append(chunk_info)

                    completed += 1
                    if completed % max(1, total // 10) == 0:
                        progress = (completed / total) * 100
                        print(f"🎵 Chunk creation progress: {completed}/{total} ({progress:.0f}%)")

                    # I/O throttling
                    if self.config.chunk_io_throttle_delay > 0:
                        time.sleep(self.config.chunk_io_throttle_delay)

                except Exception as e:
                    logger.error(f"Chunk creation failed: {e}")

        # Sort chunks by index to maintain order
        chunks.sort(key=lambda x: x["index"])
        return chunks

    def _create_chunks_sequential(self, audio: AudioSegment, chunk_params: List[Dict]) -> List[Dict]:
        """Create chunks sequentially"""
        chunks = []

        for i, params in enumerate(chunk_params):
            chunk_info = self._create_single_chunk(audio, params)
            if chunk_info:
                chunks.append(chunk_info)

            # Progress reporting
            if (i + 1) % max(1, len(chunk_params) // 10) == 0:
                progress = ((i + 1) / len(chunk_params)) * 100
                print(f"🎵 Chunk creation progress: {i+1}/{len(chunk_params)} ({progress:.0f}%)")

        return chunks

    def _create_single_chunk(self, audio: AudioSegment, params: Dict) -> Optional[Dict]:
        """Create a single audio chunk"""
        try:
            chunk = audio[params["start"]:params["end"]]

            # Save chunk to temporary file
            temp_chunk_path = f"/tmp/chunk_{params['index']}.wav"
            chunk.export(temp_chunk_path, format="wav")

            return {
                "path": temp_chunk_path,
                "start_time": params["start_time"],
                "end_time": params["end_time"],
                "index": params["index"]
            }

        except Exception as e:
            logger.error(f"Failed to create chunk {params['index']}: {e}")
            return None

    async def _process_chunks_batch_enhanced(self, chunks: List[Dict]) -> Tuple[List, List, List]:
        """Enhanced batch processing with AMP and memory optimization"""
        all_words = []
        all_segments = []
        full_text_parts = []

        total_chunks = len(chunks)
        processed_chunks = 0

        # Start performance monitoring
        self.performance_monitor.start_monitoring()

        print(f"🎤 Transcribing {total_chunks} chunks with enhanced GPU optimization...")

        try:
            if self.device == "cuda" and len(chunks) > 1:
                # GPU batch processing with AMP
                await self._process_gpu_batches(chunks, all_words, all_segments, full_text_parts)
            else:
                # Sequential processing for CPU/MPS or single chunk
                await self._process_sequential_enhanced(chunks, all_words, all_segments, full_text_parts)

        finally:
            # Stop monitoring and cleanup
            self.performance_monitor.stop_monitoring()
            self._cleanup_memory(force=True)

        return all_words, all_segments, full_text_parts

    async def _process_gpu_batches(self, chunks: List[Dict], all_words: List,
                                 all_segments: List, full_text_parts: List):
        """Process chunks in GPU batches with AMP support"""
        total_chunks = len(chunks)
        processed_chunks = 0

        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            batch_start_time = time.time()

            try:
                # Process batch with AMP context
                with self._amp_context() as amp_enabled:
                    batch_paths = [chunk["path"] for chunk in batch_chunks]

                    # Suppress NeMo output
                    with open(os.devnull, 'w') as devnull:
                        with redirect_stdout(devnull), redirect_stderr(devnull):
                            batch_outputs = self.model.transcribe(audio=batch_paths, timestamps=True)

                    # Process results
                    for chunk_info, output in zip(batch_chunks, batch_outputs):
                        self._process_chunk_result(output, chunk_info, all_words, all_segments, full_text_parts)
                        processed_chunks += 1

                # Record performance metrics
                batch_time = time.time() - batch_start_time
                self.performance_monitor.record_batch_processing(
                    len(batch_chunks), batch_time, amp_enabled, self.device
                )

                # Progress reporting
                percentage = (processed_chunks / total_chunks) * 100
                print(f"🎤 Progress: {processed_chunks}/{total_chunks} chunks ({percentage:.0f}%) "
                      f"[Batch: {batch_time:.1f}s]")

                # Memory cleanup between batches
                if processed_chunks % (self.batch_size * 2) == 0:
                    self._cleanup_memory()

            except Exception as e:
                logger.error(f"GPU batch processing failed: {e}")
                # Fallback to individual processing
                await self._process_batch_fallback(batch_chunks, all_words, all_segments, full_text_parts)
                processed_chunks += len(batch_chunks)

            finally:
                # Cleanup batch files
                for chunk_info in batch_chunks:
                    try:
                        os.remove(chunk_info["path"])
                    except:
                        pass

    async def _process_sequential_enhanced(self, chunks: List[Dict], all_words: List,
                                         all_segments: List, full_text_parts: List):
        """Enhanced sequential processing for CPU/MPS"""
        total_chunks = len(chunks)

        for i, chunk_info in enumerate(chunks):
            chunk_start_time = time.time()

            try:
                with self._amp_context() as amp_enabled:
                    await self._process_single_chunk_enhanced(chunk_info, all_words, all_segments, full_text_parts)

                # Record performance
                chunk_time = time.time() - chunk_start_time
                self.performance_monitor.record_batch_processing(1, chunk_time, amp_enabled, self.device)

                # Progress reporting
                percentage = ((i + 1) / total_chunks) * 100
                print(f"🎤 Progress: {i+1}/{total_chunks} chunks ({percentage:.0f}%) "
                      f"[Chunk: {chunk_time:.1f}s]")

                # Memory cleanup every few chunks
                if (i + 1) % 5 == 0:
                    self._cleanup_memory()

            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")

            finally:
                # Cleanup chunk file
                try:
                    os.remove(chunk_info["path"])
                except:
                    pass

    async def _process_batch_fallback(self, batch_chunks: List[Dict], all_words: List,
                                    all_segments: List, full_text_parts: List):
        """Fallback processing for failed batches"""
        for chunk_info in batch_chunks:
            try:
                await self._process_single_chunk_enhanced(chunk_info, all_words, all_segments, full_text_parts)
            except Exception as e:
                logger.error(f"Fallback processing failed for chunk: {e}")

    async def _process_single_chunk_enhanced(self, chunk_info: Dict, all_words: List,
                                           all_segments: List, full_text_parts: List):
        """Enhanced single chunk processing with error handling"""
        try:
            # Suppress NeMo's output
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    output = self.model.transcribe(audio=[chunk_info["path"]], timestamps=True)

            if output and len(output) > 0:
                self._process_chunk_result(output[0], chunk_info, all_words, all_segments, full_text_parts)

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_info.get('index', 'unknown')}: {e}")

    def _process_chunk_result(self, output: Dict, chunk_info: Dict, all_words: List,
                            all_segments: List, full_text_parts: List):
        """Process transcription output from a chunk"""
        try:
            if not output:
                return

            # Extract text
            text = output.get("text", "")
            if text:
                full_text_parts.append(text)

            # Process words with timestamps
            words = output.get("words", [])
            for word_info in words:
                if isinstance(word_info, dict):
                    start_time = word_info.get("start_time", 0) + chunk_info["start_time"]
                    end_time = word_info.get("end_time", 0) + chunk_info["start_time"]
                    word = word_info.get("word", "")

                    all_words.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "word": word
                    })

            # Process segments
            segments = output.get("segments", [])
            for segment in segments:
                if isinstance(segment, dict):
                    start_time = segment.get("start_time", 0) + chunk_info["start_time"]
                    end_time = segment.get("end_time", 0) + chunk_info["start_time"]
                    text = segment.get("text", "")

                    all_segments.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text
                    })

        except Exception as e:
            logger.error(f"Error processing chunk result: {e}")

    async def transcribe_long_audio(self, audio_file_path: str) -> Dict:
        """Enhanced transcription with comprehensive GPU optimization"""
        try:
            # Get audio info
            audio_info = AudioSegment.from_file(audio_file_path)
            duration_seconds = len(audio_info) / 1000.0

            logger.info(f"Starting enhanced transcription: {duration_seconds/3600:.2f}h audio")
            logger.info(f"Device: {self.device}, Batch size: {self.batch_size}, "
                       f"AMP: {'enabled' if self.scaler else 'disabled'}")

            # Create chunks with parallel processing
            chunks = self._create_audio_chunks_parallel(audio_file_path, duration_seconds)

            # Process chunks with enhanced optimization
            all_words, all_segments, full_text_parts = await self._process_chunks_batch_enhanced(chunks)

            # Combine results
            full_text = " ".join(full_text_parts)

            # Get performance summary
            performance_summary = self.performance_monitor.get_performance_summary()

            logger.info(f"✅ Enhanced transcription completed: {len(all_words)} words, "
                       f"{len(all_segments)} segments")
            logger.info(f"📊 Performance: {performance_summary['total_processing_time']:.1f}s, "
                       f"Peak memory: {performance_summary['peak_memory_gb']:.2f}GB, "
                       f"AMP usage: {performance_summary['amp_usage_percentage']:.1f}%")

            # Save performance metrics if enabled
            if self.config.save_performance_metrics:
                self._save_performance_metrics(performance_summary, audio_file_path)

            return {
                "text": full_text,
                "words": all_words,
                "segments": all_segments,
                "performance_metrics": performance_summary
            }

        except Exception as e:
            logger.error(f"Enhanced transcription failed: {e}")
            raise

    def _save_performance_metrics(self, metrics: Dict, audio_file_path: str):
        """Save performance metrics to file"""
        try:
            metrics_file = Path("/tmp/transcription_performance_metrics.json")

            # Load existing metrics or create new
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []

            # Add current metrics
            metrics_entry = {
                "timestamp": time.time(),
                "audio_file": Path(audio_file_path).name,
                "device": self.device,
                "batch_size": self.batch_size,
                "amp_enabled": self.scaler is not None,
                **metrics
            }

            all_metrics.append(metrics_entry)

            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)

            logger.info(f"Performance metrics saved to {metrics_file}")

        except Exception as e:
            logger.warning(f"Failed to save performance metrics: {e}")

    def cleanup_gpu_resources(self):
        """Comprehensive GPU resource cleanup"""
        try:
            print("🧹 Cleaning up enhanced GPU resources...", flush=True)

            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()

            # Close thread pool
            if self.chunk_executor:
                self.chunk_executor.shutdown(wait=True)
                self.chunk_executor = None

            # Clear model from memory
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                print("✅ Model cleared from memory", flush=True)

            # Clear AMP scaler
            if self.scaler:
                del self.scaler
                self.scaler = None

            # Force memory cleanup
            self._cleanup_memory(force=True)

            print("✅ Enhanced GPU cleanup completed", flush=True)

        except Exception as e:
            logger.error(f"GPU cleanup failed: {e}")

    def get_optimization_status(self) -> Dict:
        """Get current optimization status and capabilities"""
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "amp_supported": self._supports_amp(),
            "amp_enabled": self.scaler is not None,
            "parallel_chunking": self.config.enable_parallel_chunking,
            "memory_optimization": self.config.enable_memory_optimization,
            "device_optimization": self.config.enable_device_optimization,
            "performance_monitoring": self.config.enable_performance_monitoring,
            "max_chunk_workers": self.config.max_chunk_workers,
            "cuda_capabilities": {
                "compute_capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else None,
                "tf32_enabled": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
            } if self.device == "cuda" else None
        }

class EnhancedParakeetGPUHandler:
    """Handler wrapper for the enhanced GPU-optimized Parakeet transcriber"""

    def __init__(self, config: Optional[GPUOptimizationConfig] = None):
        self.config = config or GPUOptimizationConfig()
        self.transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=self.config)
        logger.info("EnhancedParakeetGPUHandler initialized")

    async def process_audio_files(self, video_id: str, audio_file_path: str,
                                output_dir: str) -> Dict:
        """Process audio files with enhanced GPU optimization"""
        try:
            # Transcribe audio
            result = await self.transcriber.transcribe_long_audio(audio_file_path)

            # Save results to files
            output_files = await self._save_transcription_results(
                result, video_id, audio_file_path, output_dir
            )

            return output_files

        except Exception as e:
            logger.error(f"Enhanced audio processing failed: {e}")
            raise

    async def _save_transcription_results(self, result: Dict, video_id: str,
                                        audio_file_path: str, output_dir: str) -> Dict:
        """Save transcription results to files"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save transcript JSON
            transcript_file = output_dir / f"{video_id}_transcript.json"
            transcript_data = {
                "text": result["text"],
                "segments": result["segments"],
                "metadata": {
                    "model": "nvidia/parakeet-tdt-0.6b-v2",
                    "enhanced_gpu_optimization": True,
                    "performance_metrics": result.get("performance_metrics", {})
                }
            }

            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)

            # Save words CSV
            words_file = output_dir / f"{video_id}_words.csv"
            with open(words_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['start_time', 'end_time', 'word'])
                for word in result["words"]:
                    writer.writerow([word["start_time"], word["end_time"], word["word"]])

            logger.info(f"Enhanced transcription results saved: {transcript_file}, {words_file}")

            return {
                "transcript_file": str(transcript_file),
                "words_file": str(words_file),
                "transcript_url": f"file://{transcript_file}",
                "transcriptWords_url": f"file://{words_file}"
            }

        except Exception as e:
            logger.error(f"Failed to save transcription results: {e}")
            raise

    def cleanup_gpu_resources(self):
        """Clean up GPU resources"""
        if hasattr(self, 'transcriber') and self.transcriber:
            self.transcriber.cleanup_gpu_resources()

    def get_optimization_status(self) -> Dict:
        """Get optimization status"""
        if hasattr(self, 'transcriber') and self.transcriber:
            return self.transcriber.get_optimization_status()
        return {}

# Diagnostic and testing functions
async def test_enhanced_gpu_performance():
    """Test enhanced GPU performance with comprehensive metrics"""
    print("🧪 ENHANCED GPU PERFORMANCE TEST")
    print("=" * 60)

    config = GPUOptimizationConfig()
    transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)

    # Display optimization status
    status = transcriber.get_optimization_status()
    print(f"🔧 Device: {status['device']}")
    print(f"🔧 Batch size: {status['batch_size']}")
    print(f"🔧 AMP enabled: {status['amp_enabled']}")
    print(f"🔧 Parallel chunking: {status['parallel_chunking']}")
    print(f"🔧 Memory optimization: {status['memory_optimization']}")

    if status['cuda_capabilities']:
        cuda_caps = status['cuda_capabilities']
        print(f"🔧 CUDA compute: {cuda_caps['compute_capability']}")
        print(f"🔧 GPU memory: {cuda_caps['memory_gb']:.1f}GB")
        print(f"🔧 TF32 enabled: {cuda_caps['tf32_enabled']}")

    print()

    # Test with sample file (if available)
    test_files = [
        "/tmp/test_audio.wav",
        "/Users/aman/Downloads/test_audio.mp3",
        "output/raw/audio/test_audio.wav"
    ]

    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = file_path
            break

    if test_file:
        print(f"🎵 Testing with: {test_file}")

        try:
            start_time = time.time()
            result = await transcriber.transcribe_long_audio(test_file)
            end_time = time.time()

            processing_time = end_time - start_time
            performance = result.get("performance_metrics", {})

            print(f"✅ Enhanced GPU Test Results:")
            print(f"   Processing time: {processing_time:.1f}s")
            print(f"   Words transcribed: {len(result['words'])}")
            print(f"   Segments: {len(result['segments'])}")
            print(f"   Peak memory: {performance.get('peak_memory_gb', 0):.2f}GB")
            print(f"   Average batch size: {performance.get('average_batch_size', 0):.1f}")
            print(f"   AMP usage: {performance.get('amp_usage_percentage', 0):.1f}%")
            print(f"   Memory efficiency: {performance.get('memory_efficiency', 0):.1f}%")
            print(f"   Text preview: {result['text'][:100]}...")

        except Exception as e:
            print(f"❌ Enhanced GPU test failed: {e}")

        finally:
            transcriber.cleanup_gpu_resources()
    else:
        print("⚠️  No test audio file found")
        print("   Create a test file at one of these locations:")
        for file_path in test_files:
            print(f"   - {file_path}")

def verify_optimization_features():
    """Verify that all optimization features are working correctly"""
    print("🔍 OPTIMIZATION FEATURE VERIFICATION")
    print("=" * 60)

    config = GPUOptimizationConfig()

    # Test 1: AMP Support
    print("1️⃣ Testing AMP Support")
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        amp_supported = compute_capability[0] >= 7
        print(f"   CUDA available: ✅")
        print(f"   Compute capability: {compute_capability[0]}.{compute_capability[1]}")
        print(f"   AMP supported: {'✅' if amp_supported else '❌'}")
        print(f"   AMP enabled in config: {'✅' if config.enable_amp else '❌'}")
    else:
        print(f"   CUDA available: ❌")
        print(f"   AMP supported: ❌")

    print()

    # Test 2: Memory Optimization
    print("2️⃣ Testing Memory Optimization")
    print(f"   Memory optimization enabled: {'✅' if config.enable_memory_optimization else '❌'}")
    print(f"   Cleanup threshold: {config.memory_cleanup_threshold}")
    print(f"   Max fragmentation: {config.max_memory_fragmentation}")

    print()

    # Test 3: Parallel Chunking
    print("3️⃣ Testing Parallel Chunking")
    print(f"   Parallel chunking enabled: {'✅' if config.enable_parallel_chunking else '❌'}")
    print(f"   Max chunk workers: {config.max_chunk_workers}")
    print(f"   CPU cores available: {multiprocessing.cpu_count()}")

    print()

    # Test 4: Device Optimizations
    print("4️⃣ Testing Device Optimizations")
    print(f"   Device optimization enabled: {'✅' if config.enable_device_optimization else '❌'}")

    if torch.cuda.is_available():
        print(f"   cuDNN benchmark: {'✅' if config.cuda_benchmark else '❌'}")
        print(f"   TF32 enabled: {'✅' if config.cuda_tf32 else '❌'}")

        # Check actual PyTorch settings
        print(f"   cuDNN benchmark (actual): {'✅' if torch.backends.cudnn.benchmark else '❌'}")
        print(f"   TF32 matmul (actual): {'✅' if torch.backends.cuda.matmul.allow_tf32 else '❌'}")
        print(f"   TF32 cuDNN (actual): {'✅' if torch.backends.cudnn.allow_tf32 else '❌'}")

    print()

    # Test 5: Performance Monitoring
    print("5️⃣ Testing Performance Monitoring")
    print(f"   Performance monitoring enabled: {'✅' if config.enable_performance_monitoring else '❌'}")
    print(f"   Monitoring interval: {config.memory_monitoring_interval}s")
    print(f"   Save metrics: {'✅' if config.save_performance_metrics else '❌'}")

    print()
    print("✅ Optimization feature verification completed")

if __name__ == "__main__":
    import asyncio

    print("🚀 ENHANCED PARAKEET GPU TRANSCRIBER")
    print("=" * 60)

    # Run verification
    verify_optimization_features()
    print()

    # Run performance test
    asyncio.run(test_enhanced_gpu_performance())
