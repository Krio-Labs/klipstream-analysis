#!/usr/bin/env python3
"""
Transcription Configuration Settings

This module manages configuration for the GPU-optimized transcription system,
including environment variables, method selection, and performance tuning.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class TranscriptionConfig:
    """Configuration settings for transcription system"""
    
    # Method Selection
    transcription_method: str = "auto"  # auto, parakeet, deepgram, hybrid
    enable_gpu_transcription: bool = True
    enable_fallback: bool = True
    cost_optimization: bool = True
    
    # GPU Configuration
    parakeet_model_name: str = "nvidia/parakeet-tdt-0.6b-v2"
    gpu_batch_size: int = 8
    gpu_memory_limit_gb: float = 20.0
    
    # Processing Parameters
    chunk_duration_minutes: int = 10
    chunk_overlap_seconds: int = 10
    max_concurrent_chunks: int = 4
    enable_batch_processing: bool = True
    
    # Decision Thresholds
    short_file_threshold_hours: float = 2.0
    long_file_threshold_hours: float = 4.0
    cost_threshold_per_hour: float = 0.10
    
    # Performance Monitoring
    enable_performance_metrics: bool = True
    log_transcription_costs: bool = True
    metrics_export_interval: int = 60
    
    # Cloud Run Specific
    cloud_run_timeout_seconds: int = 3600
    max_file_size_gb: float = 2.0
    
    @classmethod
    def from_environment(cls) -> 'TranscriptionConfig':
        """Create configuration from environment variables"""
        
        return cls(
            # Method Selection
            transcription_method=os.getenv("TRANSCRIPTION_METHOD", "auto"),
            enable_gpu_transcription=os.getenv("ENABLE_GPU_TRANSCRIPTION", "true").lower() == "true",
            enable_fallback=os.getenv("ENABLE_FALLBACK", "true").lower() == "true",
            cost_optimization=os.getenv("COST_OPTIMIZATION", "true").lower() == "true",
            
            # GPU Configuration
            parakeet_model_name=os.getenv("PARAKEET_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2"),
            gpu_batch_size=int(os.getenv("GPU_BATCH_SIZE", "8")),
            gpu_memory_limit_gb=float(os.getenv("GPU_MEMORY_LIMIT_GB", "20.0")),
            
            # Processing Parameters
            chunk_duration_minutes=int(os.getenv("CHUNK_DURATION_MINUTES", "10")),
            chunk_overlap_seconds=int(os.getenv("CHUNK_OVERLAP_SECONDS", "10")),
            max_concurrent_chunks=int(os.getenv("MAX_CONCURRENT_CHUNKS", "4")),
            enable_batch_processing=os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true",
            
            # Decision Thresholds
            short_file_threshold_hours=float(os.getenv("SHORT_FILE_THRESHOLD_HOURS", "2.0")),
            long_file_threshold_hours=float(os.getenv("LONG_FILE_THRESHOLD_HOURS", "4.0")),
            cost_threshold_per_hour=float(os.getenv("COST_THRESHOLD_PER_HOUR", "0.10")),
            
            # Performance Monitoring
            enable_performance_metrics=os.getenv("ENABLE_PERFORMANCE_METRICS", "true").lower() == "true",
            log_transcription_costs=os.getenv("LOG_TRANSCRIPTION_COSTS", "true").lower() == "true",
            metrics_export_interval=int(os.getenv("METRICS_EXPORT_INTERVAL", "60")),
            
            # Cloud Run Specific
            cloud_run_timeout_seconds=int(os.getenv("CLOUD_RUN_TIMEOUT_SECONDS", "3600")),
            max_file_size_gb=float(os.getenv("MAX_FILE_SIZE_GB", "2.0"))
        )
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration settings"""
        
        issues = []
        
        # Validate method selection
        valid_methods = ["auto", "parakeet", "parakeet_enhanced", "deepgram", "hybrid"]
        if self.transcription_method not in valid_methods:
            issues.append(f"Invalid transcription_method: {self.transcription_method}")
        
        # Validate GPU settings
        if self.gpu_batch_size < 1 or self.gpu_batch_size > 16:
            issues.append(f"GPU batch size should be 1-16, got {self.gpu_batch_size}")
        
        if self.gpu_memory_limit_gb < 4 or self.gpu_memory_limit_gb > 80:
            issues.append(f"GPU memory limit should be 4-80GB, got {self.gpu_memory_limit_gb}")
        
        # Validate processing parameters
        if self.chunk_duration_minutes < 1 or self.chunk_duration_minutes > 30:
            issues.append(f"Chunk duration should be 1-30 minutes, got {self.chunk_duration_minutes}")
        
        if self.chunk_overlap_seconds < 0 or self.chunk_overlap_seconds > 60:
            issues.append(f"Chunk overlap should be 0-60 seconds, got {self.chunk_overlap_seconds}")
        
        # Validate thresholds
        if self.short_file_threshold_hours >= self.long_file_threshold_hours:
            issues.append("Short file threshold should be less than long file threshold")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config": self.__dict__
        }
    
    def get_method_for_duration(self, duration_hours: float, gpu_available: bool = True) -> str:
        """Get recommended transcription method for given duration"""
        
        if self.transcription_method != "auto":
            return self.transcription_method
        
        if not gpu_available:
            if duration_hours < self.short_file_threshold_hours:
                return "deepgram"  # Fast for short files
            else:
                return "deepgram"  # Reliable for long files
        
        # GPU available - prefer enhanced version if enabled
        enhanced_enabled = os.getenv("ENABLE_ENHANCED_GPU_OPTIMIZATION", "true").lower() == "true"
        parakeet_method = "parakeet_enhanced" if enhanced_enabled else "parakeet"

        if duration_hours < self.short_file_threshold_hours:
            return parakeet_method
        elif duration_hours < self.long_file_threshold_hours:
            return "hybrid"
        else:
            return "deepgram"  # Avoid timeout risk
    
    def get_chunk_parameters(self, gpu_available: bool = True) -> Dict[str, int]:
        """Get optimal chunk parameters based on configuration"""
        
        if gpu_available:
            # GPU can handle larger chunks
            chunk_duration = self.chunk_duration_minutes
            batch_size = self.gpu_batch_size
        else:
            # CPU needs smaller chunks
            chunk_duration = max(3, self.chunk_duration_minutes // 2)
            batch_size = max(1, self.gpu_batch_size // 4)
        
        return {
            "chunk_duration_seconds": chunk_duration * 60,
            "overlap_seconds": self.chunk_overlap_seconds,
            "batch_size": batch_size,
            "max_concurrent": self.max_concurrent_chunks
        }

# Global configuration instance
config = TranscriptionConfig.from_environment()

# Validation on import (only warn, don't fail)
try:
    validation_result = config.validate()
    if not validation_result["valid"]:
        import warnings
        warnings.warn(f"Configuration validation issues: {validation_result['issues']}")
except Exception as e:
    import warnings
    warnings.warn(f"Configuration validation failed: {e}")

def get_config() -> TranscriptionConfig:
    """Get the global configuration instance"""
    return config

def reload_config() -> TranscriptionConfig:
    """Reload configuration from environment variables"""
    global config
    config = TranscriptionConfig.from_environment()
    return config
