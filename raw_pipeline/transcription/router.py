#!/usr/bin/env python3
"""
Transcription Router

Central orchestrator for intelligent transcription method selection and execution.
Handles GPU Parakeet, Deepgram, and hybrid processing with automatic fallback.
"""

import asyncio
import time
import os
from typing import Dict, Optional, Tuple
from pathlib import Path

from utils.logging_setup import setup_logger
from .config.settings import get_config
from .handlers.deepgram_handler import DeepgramHandler
from .utils.fallback_manager import FallbackManager
from .utils.cost_optimizer import CostOptimizer

# Import audio utils with fallback
try:
    from utils.audio_utils import get_audio_info, validate_audio_file
except ImportError:
    # Fallback implementations for testing
    def get_audio_info(audio_file_path: str) -> Dict:
        """Fallback audio info function"""
        import os
        file_size = os.path.getsize(audio_file_path) if os.path.exists(audio_file_path) else 0
        return {
            "duration_seconds": 3600,  # Default 1 hour
            "duration_minutes": 60,
            "duration_hours": 1.0,
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "sample_rate": 16000,
            "channels": 1
        }

    def validate_audio_file(audio_file_path: str) -> Tuple[bool, str]:
        """Fallback audio validation function"""
        import os
        if not os.path.exists(audio_file_path):
            return False, "File does not exist"
        return True, "Valid audio file"

logger = setup_logger("transcription_router", "transcription_router.log")

class TranscriptionRouter:
    """Central orchestrator for transcription method selection and execution"""
    
    def __init__(self):
        # Always reload config to respect environment changes
        from .config.settings import reload_config
        self.config = reload_config()

        self.deepgram_handler = DeepgramHandler()
        self.parakeet_handler = None  # Lazy load
        self.hybrid_processor = None  # Lazy load
        self.fallback_manager = FallbackManager()
        self.cost_optimizer = CostOptimizer()

        # Log configuration status
        force_deepgram = os.environ.get("FORCE_DEEPGRAM_TRANSCRIPTION", "").lower() == "true"
        logger.info(f"TranscriptionRouter initialized - Force Deepgram: {force_deepgram}")
        logger.info(f"Config method: {self.config.transcription_method}, GPU enabled: {self.config.enable_gpu_transcription}")
    
    def _lazy_load_parakeet(self):
        """Lazy load Parakeet handler to avoid import issues when GPU not available"""
        if self.parakeet_handler is None:
            try:
                # Check if enhanced GPU optimization is enabled
                use_enhanced = os.environ.get("ENABLE_ENHANCED_GPU_OPTIMIZATION", "true").lower() == "true"

                if use_enhanced:
                    try:
                        from .handlers.enhanced_parakeet_gpu import EnhancedParakeetGPUHandler
                        self.parakeet_handler = EnhancedParakeetGPUHandler()
                        logger.info("Enhanced Parakeet GPU handler loaded successfully")
                    except Exception as enhanced_error:
                        logger.warning(f"Enhanced Parakeet GPU handler failed: {enhanced_error}")
                        logger.info("Falling back to standard Parakeet GPU handler")
                        from .handlers.parakeet_gpu import ParakeetGPUHandler
                        self.parakeet_handler = ParakeetGPUHandler()
                        logger.info("Standard Parakeet GPU handler loaded successfully")
                else:
                    from .handlers.parakeet_gpu import ParakeetGPUHandler
                    self.parakeet_handler = ParakeetGPUHandler()
                    logger.info("Standard Parakeet GPU handler loaded successfully")

            except Exception as e:
                logger.warning(f"Failed to load any Parakeet GPU handler: {e}")
                self.parakeet_handler = False  # Mark as failed
        return self.parakeet_handler if self.parakeet_handler is not False else None
    
    def _lazy_load_hybrid(self):
        """Lazy load hybrid processor"""
        if self.hybrid_processor is None:
            try:
                from .handlers.hybrid_processor import HybridProcessor
                self.hybrid_processor = HybridProcessor()
                logger.info("Hybrid processor loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load hybrid processor: {e}")
                self.hybrid_processor = False
        return self.hybrid_processor if self.hybrid_processor is not False else None
    
    async def transcribe(self, audio_file_path: str, video_id: str, output_dir: str = None) -> Dict:
        """
        Main transcription entry point with intelligent method selection
        
        Args:
            audio_file_path (str): Path to audio file
            video_id (str): Video identifier
            output_dir (str): Output directory for results
            
        Returns:
            Dict: Transcription results with file paths
        """
        start_time = time.time()
        
        try:
            # Validate audio file
            is_valid, error_msg = validate_audio_file(audio_file_path)
            if not is_valid:
                logger.error(f"Audio validation failed: {error_msg}")
                return {"error": error_msg, "status": "failed"}
            
            # Get audio information
            audio_info = get_audio_info(audio_file_path)
            if not audio_info:
                logger.error("Failed to get audio information")
                return {"error": "Failed to analyze audio file", "status": "failed"}
            
            duration_hours = audio_info["duration_hours"]
            file_size_mb = audio_info["file_size_mb"]
            
            logger.info(f"Processing audio: {duration_hours:.2f}h, {file_size_mb:.1f}MB")
            
            # Select optimal transcription method
            method = self._select_transcription_method(audio_info)
            logger.info(f"Selected transcription method: {method}")
            
            # Execute transcription with fallback
            result = await self._execute_transcription(
                method, audio_file_path, video_id, output_dir, audio_info
            )
            
            # Calculate and log performance metrics
            total_time = time.time() - start_time
            self._log_performance_metrics(method, audio_info, result, total_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return await self.fallback_manager.handle_complete_failure(
                e, audio_file_path, video_id, output_dir
            )
    
    def _select_transcription_method(self, audio_info: Dict) -> str:
        """Select optimal transcription method based on audio characteristics"""

        # PRIORITY 1: Check for forced Deepgram transcription (temporary override)
        force_deepgram = os.environ.get("FORCE_DEEPGRAM_TRANSCRIPTION", "").lower() == "true"
        if force_deepgram:
            logger.info("🔄 FORCED: Using Deepgram transcription (temporary override)")
            return "deepgram"

        # PRIORITY 2: Check if transcription method is explicitly set to deepgram
        if self.config.transcription_method == "deepgram":
            logger.info("🌐 Using Deepgram transcription (explicitly configured)")
            return "deepgram"

        # PRIORITY 3: Check if GPU transcription is disabled
        if not self.config.enable_gpu_transcription:
            logger.info("🔄 GPU transcription disabled, using Deepgram")
            return "deepgram"

        duration_hours = audio_info["duration_hours"]
        file_size_mb = audio_info["file_size_mb"]

        # Check if GPU is available
        gpu_available = self._is_gpu_available()

        # Use configuration-based method selection
        method = self.config.get_method_for_duration(duration_hours, gpu_available)

        # Cost optimization override
        if self.config.cost_optimization:
            optimal_method = self.cost_optimizer.get_optimal_method(
                duration_hours, gpu_available, file_size_mb
            )
            if optimal_method != method:
                logger.info(f"Cost optimization: {method} → {optimal_method}")
                method = optimal_method

        # Final validation
        if method in ["parakeet", "parakeet_enhanced"] and not gpu_available:
            logger.warning("Parakeet selected but GPU unavailable, falling back to Deepgram")
            method = "deepgram"

        return method
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for transcription (NVIDIA CUDA or Apple Metal)"""

        # Check for NVIDIA CUDA
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("NVIDIA CUDA GPU detected for transcription")
                return True
            else:
                logger.debug("PyTorch available but CUDA not detected")
        except ImportError:
            logger.debug("PyTorch not available for CUDA detection")

        # Check for Apple Silicon Metal Performance Shaders
        try:
            import platform
            if platform.system() == "Darwin":  # macOS
                import subprocess
                result = subprocess.run(["system_profiler", "SPHardwareDataType"],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    output = result.stdout
                    if "Apple" in output and any(chip in output for chip in ["M1", "M2", "M3", "M4"]):
                        logger.info("Apple Silicon Metal GPU detected for transcription")
                        return True
                    else:
                        logger.debug(f"macOS detected but no Apple Silicon found in: {output[:200]}...")
                else:
                    logger.debug(f"system_profiler failed with return code: {result.returncode}")
        except Exception as e:
            logger.warning(f"Apple Silicon detection failed: {e}")

        # Check environment variable override (for consistency with main.py)
        if os.environ.get("TRANSCRIPTION_METHOD") == "parakeet":
            logger.info("Parakeet transcription forced via environment variable")
            return True

        logger.warning("No GPU acceleration available for transcription - falling back to Deepgram")
        return False
    
    async def _execute_transcription(self, method: str, audio_file_path: str, 
                                   video_id: str, output_dir: str, audio_info: Dict) -> Dict:
        """Execute transcription using selected method with fallback"""
        
        try:
            if method in ["parakeet", "parakeet_gpu", "parakeet_enhanced"]:
                return await self._execute_parakeet_transcription(
                    audio_file_path, video_id, output_dir
                )
            elif method == "hybrid":
                return await self._execute_hybrid_transcription(
                    audio_file_path, video_id, output_dir, audio_info
                )
            elif method == "deepgram":
                return await self._execute_deepgram_transcription(
                    audio_file_path, video_id, output_dir
                )
            else:
                raise ValueError(f"Unknown transcription method: {method}")

        except Exception as e:
            logger.error(f"Transcription method {method} failed: {e}")
            
            if self.config.enable_fallback:
                return await self.fallback_manager.handle_method_failure(
                    method, e, audio_file_path, video_id, output_dir
                )
            else:
                raise
    
    async def _execute_parakeet_transcription(self, audio_file_path: str, 
                                            video_id: str, output_dir: str) -> Dict:
        """Execute Parakeet GPU transcription"""
        
        parakeet = self._lazy_load_parakeet()
        if not parakeet:
            raise RuntimeError("Parakeet GPU handler not available")
        
        return await parakeet.process_audio_files(
            video_id=video_id,
            audio_file_path=audio_file_path,
            output_dir=output_dir
        )
    
    async def _execute_hybrid_transcription(self, audio_file_path: str, 
                                          video_id: str, output_dir: str, 
                                          audio_info: Dict) -> Dict:
        """Execute hybrid transcription (Parakeet + Deepgram)"""
        
        hybrid = self._lazy_load_hybrid()
        if not hybrid:
            # Fallback to Parakeet or Deepgram
            logger.warning("Hybrid processor not available, falling back to Parakeet")
            return await self._execute_parakeet_transcription(
                audio_file_path, video_id, output_dir
            )
        
        return await hybrid.process_audio_files(
            video_id=video_id,
            audio_file_path=audio_file_path,
            output_dir=output_dir,
            audio_info=audio_info
        )
    
    async def _execute_deepgram_transcription(self, audio_file_path: str, 
                                            video_id: str, output_dir: str) -> Dict:
        """Execute Deepgram transcription"""
        
        return await self.deepgram_handler.process_audio_files(
            video_id=video_id,
            audio_file_path=audio_file_path,
            output_dir=output_dir
        )
    
    def _log_performance_metrics(self, method: str, audio_info: Dict, 
                               result: Dict, total_time: float):
        """Log performance metrics for monitoring"""
        
        if not self.config.enable_performance_metrics:
            return
        
        duration_hours = audio_info["duration_hours"]
        file_size_mb = audio_info["file_size_mb"]
        
        # Calculate performance metrics
        processing_speed_ratio = (duration_hours * 3600) / total_time if total_time > 0 else 0
        
        # Calculate cost
        cost = self.cost_optimizer.calculate_transcription_cost(
            duration_hours * 3600, method
        )
        
        # Log metrics
        metrics = {
            "method": method,
            "duration_hours": duration_hours,
            "file_size_mb": file_size_mb,
            "processing_time_seconds": total_time,
            "processing_speed_ratio": processing_speed_ratio,
            "estimated_cost": cost,
            "success": "error" not in result,
            "timestamp": time.time()
        }
        
        logger.info(f"Performance metrics: {metrics}")
        
        # Export to cost optimizer for tracking
        self.cost_optimizer.record_transcription(
            method, duration_hours * 3600, total_time, cost
        )

# Backward compatibility wrapper
class TranscriptionHandler(TranscriptionRouter):
    """Backward compatibility wrapper for existing code"""
    
    async def process_audio_files(self, video_id: str, audio_file_path: str, 
                                output_dir: str = None) -> Dict:
        """Legacy interface compatibility"""
        return await self.transcribe(audio_file_path, video_id, output_dir)
