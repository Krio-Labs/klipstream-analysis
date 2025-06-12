"""
Pipeline Wrapper Service

This module provides enhanced integration with the existing pipeline,
adding progress tracking, error handling, and retry mechanisms.
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

from ..models import ProcessingStage
from .retry_manager import retry_network_operation, retry_processing_operation
from .error_handler import error_classifier

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    """Progress information for pipeline stages"""
    stage: ProcessingStage
    percentage: float
    message: str
    details: Optional[Dict[str, Any]] = None


class PipelineProgressTracker:
    """
    Enhanced progress tracker for the video analysis pipeline
    """
    
    def __init__(self, job_id: str, progress_callback: Optional[Callable] = None):
        self.job_id = job_id
        self.progress_callback = progress_callback
        self.stage_weights = {
            ProcessingStage.QUEUED: (0, 0),
            ProcessingStage.DOWNLOADING: (0, 20),
            ProcessingStage.GENERATING_WAVEFORM: (20, 25),
            ProcessingStage.TRANSCRIBING: (25, 50),
            ProcessingStage.FETCHING_CHAT: (50, 60),
            ProcessingStage.ANALYZING: (60, 85),
            ProcessingStage.FINDING_HIGHLIGHTS: (85, 95),
            ProcessingStage.COMPLETED: (100, 100)
        }
        self.current_stage = ProcessingStage.QUEUED
        self.stage_start_time = datetime.utcnow()
        
    async def update_progress(self, stage: ProcessingStage, sub_progress: float = 0.0, message: str = None):
        """Update progress for a specific stage"""
        if stage != self.current_stage:
            self.current_stage = stage
            self.stage_start_time = datetime.utcnow()
        
        # Calculate overall progress
        start_percent, end_percent = self.stage_weights.get(stage, (0, 0))
        stage_range = end_percent - start_percent
        stage_contribution = (sub_progress / 100.0) * stage_range
        overall_progress = start_percent + stage_contribution
        
        progress_info = PipelineProgress(
            stage=stage,
            percentage=overall_progress,
            message=message or f"Processing: {stage.value}",
            details={
                "stage_progress": sub_progress,
                "stage_start_time": self.stage_start_time.isoformat(),
                "elapsed_in_stage": (datetime.utcnow() - self.stage_start_time).total_seconds()
            }
        )
        
        logger.debug(f"Job {self.job_id}: {stage.value} - {overall_progress:.1f}% ({message})")
        
        if self.progress_callback:
            try:
                await self.progress_callback(stage, overall_progress, message)
            except Exception as e:
                logger.error(f"Error in progress callback for job {self.job_id}: {str(e)}")


class EnhancedPipelineWrapper:
    """
    Wrapper for the existing pipeline with enhanced features
    """
    
    def __init__(self):
        logger.info("EnhancedPipelineWrapper initialized")
    
    async def run_integrated_pipeline_with_tracking(
        self,
        video_url: str,
        job_id: str,
        progress_callback: Optional[Callable] = None,
        transcription_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run the integrated pipeline with enhanced progress tracking and error handling

        UPDATED: Now fully compatible with main.py pipeline including:
        - Proper transcription configuration
        - URL updates for all 7 Convex fields
        - Enhanced error handling and status tracking
        - GPU optimization support

        Args:
            video_url: URL of the video to process
            job_id: Unique job identifier
            progress_callback: Callback function for progress updates
            transcription_config: Transcription configuration (method, gpu, fallback, etc.)

        Returns:
            Dictionary with processing results including all file URLs
        """
        tracker = PipelineProgressTracker(job_id, progress_callback)

        try:
            logger.info(f"🚀 Starting enhanced pipeline for job {job_id}")
            logger.info(f"📹 Video URL: {video_url}")

            # Stage 1: Queue and initialize
            await tracker.update_progress(ProcessingStage.QUEUED, 0, "Initializing analysis...")

            # Configure transcription settings if provided
            original_env = {}
            if transcription_config:
                logger.info(f"🎤 Applying transcription configuration: {transcription_config}")
                original_env = self._apply_transcription_config(transcription_config)
            else:
                logger.info("🎤 Using default transcription configuration")

            # Import the main pipeline function (updated to use current main.py)
            from main import run_integrated_pipeline

            # Create a wrapper function that properly handles the async pipeline
            async def pipeline_execution():
                """Execute the main pipeline with proper async handling"""
                try:
                    # The main pipeline is async, so we await it directly
                    logger.info(f"🔄 Executing main.py pipeline for job {job_id}")
                    result = await run_integrated_pipeline(video_url)

                    # Log pipeline result for debugging
                    if result:
                        logger.info(f"✅ Pipeline result status: {result.get('status', 'unknown')}")
                        if result.get('video_id'):
                            logger.info(f"📹 Video ID: {result['video_id']}")
                        if result.get('stage_times'):
                            logger.info(f"⏱️ Stage times: {result['stage_times']}")
                        if result.get('transcription_metadata'):
                            logger.info(f"🎤 Transcription metadata: {result['transcription_metadata']}")
                    else:
                        logger.warning("⚠️ Pipeline returned no result")

                    return result

                except Exception as e:
                    logger.error(f"❌ Pipeline execution failed: {str(e)}")
                    raise

            # Stage 2: Start pipeline execution
            await tracker.update_progress(ProcessingStage.DOWNLOADING, 5, "Starting pipeline execution...")

            # Execute pipeline with retry logic
            result = await retry_processing_operation(
                pipeline_execution,
                operation_name=f"pipeline_execution_{job_id}"
            )

            # Enhanced result validation and processing
            if result and result.get("status") == "completed":
                # Extract comprehensive results from pipeline
                video_id = result.get("video_id")
                files = result.get("files", {})
                stage_times = result.get("stage_times", {})
                transcription_metadata = result.get("transcription_metadata", {})
                twitch_info = result.get("twitch_info", {})

                # Log successful completion with details
                logger.info(f"✅ Pipeline completed successfully for job {job_id}")
                logger.info(f"📹 Video ID: {video_id}")
                logger.info(f"📁 Files processed: {len(files)} files")
                logger.info(f"⏱️ Total duration: {result.get('total_duration', 0):.1f}s")

                # Log transcription details if available
                if transcription_metadata:
                    method_used = transcription_metadata.get("method_used", "unknown")
                    cost_estimate = transcription_metadata.get("cost_estimate", 0.0)
                    gpu_used = transcription_metadata.get("gpu_used", False)
                    processing_time = transcription_metadata.get("processing_time_seconds", 0.0)

                    logger.info(f"🎤 Transcription method: {method_used}")
                    logger.info(f"💰 Estimated cost: ${cost_estimate:.3f}")
                    logger.info(f"🖥️ GPU used: {'Yes' if gpu_used else 'No'}")
                    logger.info(f"⏱️ Transcription time: {processing_time:.1f}s")

                # Update final progress
                await tracker.update_progress(ProcessingStage.COMPLETED, 100, "Analysis completed successfully!")

                # Return enhanced result with all metadata
                enhanced_result = {
                    **result,
                    "job_id": job_id,
                    "api_version": "2.0.0",
                    "enhanced_tracking": True
                }

                return enhanced_result

            else:
                # Pipeline failed or returned unexpected result
                error_message = result.get("error", "Unknown error") if result else "Pipeline returned no result"
                logger.error(f"❌ Pipeline failed for job {job_id}: {error_message}")

                # Create detailed error response
                error_result = {
                    "status": "failed",
                    "error": error_message,
                    "job_id": job_id,
                    "stage_times": result.get("stage_times", {}) if result else {},
                    "total_duration": result.get("total_duration", 0) if result else 0
                }

                raise Exception(f"Pipeline execution failed: {error_message}")

        except Exception as e:
            logger.error(f"❌ Enhanced pipeline failed for job {job_id}: {str(e)}")

            # Update progress with failure
            await tracker.update_progress(
                ProcessingStage.FAILED,
                tracker.stage_weights[tracker.current_stage][0],  # Keep current progress
                f"Failed: {str(e)}"
            )

            # Re-raise with context
            raise Exception(f"Enhanced pipeline failed: {str(e)}") from e
            
        except Exception as e:
            logger.error(f"Pipeline failed for job {job_id}: {str(e)}")

            # Enhanced error classification with transcription context
            error_context = {
                "job_id": job_id,
                "video_url": video_url,
                "current_stage": tracker.current_stage.value,
                "stage_elapsed_time": (datetime.utcnow() - tracker.stage_start_time).total_seconds()
            }

            # Add transcription context if available
            if transcription_config:
                error_context.update({
                    "transcription_method": transcription_config.get("method", "auto"),
                    "gpu_enabled": transcription_config.get("enable_gpu", True),
                    "cost_optimization": transcription_config.get("cost_optimization", True),
                    "fallback_enabled": transcription_config.get("enable_fallback", True)
                })

            # Use specialized transcription error classification if in transcription stage
            if tracker.current_stage == ProcessingStage.TRANSCRIBING and transcription_config:
                error_info = error_classifier.classify_transcription_error(e, error_context)
            else:
                error_info = error_classifier.classify_error(e, error_context)
            
            # Update progress with failure
            await tracker.update_progress(
                ProcessingStage.FAILED,
                tracker.stage_weights[tracker.current_stage][0],  # Keep current progress
                f"Failed: {error_info.error_message}"
            )
            
            # Re-raise with enhanced error info
            raise Exception(f"Pipeline failed: {error_info.error_message}") from e

        finally:
            # Restore original environment variables
            if transcription_config and original_env:
                self._restore_environment(original_env)
    
    async def run_raw_pipeline_with_tracking(
        self,
        video_url: str,
        job_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run just the raw pipeline (download, transcription) with tracking
        """
        tracker = PipelineProgressTracker(job_id, progress_callback)
        
        try:
            await tracker.update_progress(ProcessingStage.DOWNLOADING, 0, "Starting raw pipeline...")

            # Import raw pipeline function
            from raw_pipeline import process_raw_files

            # Create progress wrapper for raw pipeline
            async def raw_pipeline_execution():
                return await process_raw_files(video_url)

            # Execute with retry
            result = await retry_network_operation(
                raw_pipeline_execution,
                operation_name=f"raw_pipeline_{job_id}"
            )

            await tracker.update_progress(ProcessingStage.TRANSCRIBING, 100, "Raw pipeline completed!")

            return result
            
        except Exception as e:
            logger.error(f"Raw pipeline failed for job {job_id}: {str(e)}")
            error_info = error_classifier.classify_error(e, {"job_id": job_id, "video_url": video_url})
            raise Exception(f"Raw pipeline failed: {error_info.error_message}") from e
    
    async def run_analysis_pipeline_with_tracking(
        self,
        video_id: str,
        job_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run just the analysis pipeline with tracking
        """
        tracker = PipelineProgressTracker(job_id, progress_callback)
        
        try:
            await tracker.update_progress(ProcessingStage.ANALYZING, 0, "Starting analysis pipeline...")

            # Import analysis pipeline function
            from analysis_pipeline import process_analysis

            # Create progress wrapper
            async def analysis_pipeline_execution():
                return await process_analysis(video_id)

            # Execute with retry
            result = await retry_processing_operation(
                analysis_pipeline_execution,
                operation_name=f"analysis_pipeline_{job_id}"
            )

            await tracker.update_progress(ProcessingStage.COMPLETED, 100, "Analysis pipeline completed!")

            return result
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed for job {job_id}: {str(e)}")
            error_info = error_classifier.classify_error(e, {"job_id": job_id, "video_id": video_id})
            raise Exception(f"Analysis pipeline failed: {error_info.error_message}") from e
    
    def get_stage_description(self, stage: ProcessingStage) -> str:
        """Get user-friendly description of processing stage"""
        descriptions = {
            ProcessingStage.QUEUED: "Your video has been queued for processing",
            ProcessingStage.DOWNLOADING: "Downloading video and extracting audio",
            ProcessingStage.GENERATING_WAVEFORM: "Generating audio waveform visualization",
            ProcessingStage.TRANSCRIBING: "Generating transcript from audio using AI",
            ProcessingStage.FETCHING_CHAT: "Downloading chat messages and metadata",
            ProcessingStage.ANALYZING: "Analyzing sentiment and content patterns",
            ProcessingStage.FINDING_HIGHLIGHTS: "Detecting highlights and key moments",
            ProcessingStage.COMPLETED: "Analysis completed successfully!",
            ProcessingStage.FAILED: "Analysis failed - please try again"
        }
        return descriptions.get(stage, f"Processing: {stage.value}")

    def _apply_transcription_config(self, transcription_config: Dict) -> Dict[str, str]:
        """
        Apply transcription configuration by setting environment variables

        UPDATED: Now fully compatible with main.py transcription system including:
        - Proper method mapping (auto, parakeet, deepgram, hybrid)
        - GPU optimization settings
        - Cost optimization and fallback configuration
        - Temporary Deepgram override support

        Args:
            transcription_config: Dictionary with transcription configuration

        Returns:
            Dictionary of original environment variable values for restoration
        """
        import os

        original_env = {}

        # Enhanced mapping for full main.py compatibility
        env_mapping = {
            'method': 'TRANSCRIPTION_METHOD',
            'enable_gpu': 'ENABLE_GPU_TRANSCRIPTION',
            'enable_fallback': 'ENABLE_FALLBACK',
            'cost_optimization': 'COST_OPTIMIZATION'
        }

        # Additional environment variables for enhanced GPU optimization
        gpu_optimization_mapping = {
            'enable_amp': 'ENABLE_AMP',
            'enable_memory_optimization': 'ENABLE_MEMORY_OPTIMIZATION',
            'enable_parallel_chunking': 'ENABLE_PARALLEL_CHUNKING',
            'enable_device_optimization': 'ENABLE_DEVICE_OPTIMIZATION',
            'enable_performance_monitoring': 'ENABLE_PERFORMANCE_MONITORING'
        }

        # Apply main transcription configuration
        for config_key, env_var in env_mapping.items():
            if config_key in transcription_config:
                # Store original value
                original_env[env_var] = os.environ.get(env_var)

                # Set new value
                value = transcription_config[config_key]
                if isinstance(value, bool):
                    value = "true" if value else "false"
                os.environ[env_var] = str(value)

                logger.info(f"🎤 Set {env_var}={value}")

        # Apply GPU optimization settings if GPU is enabled
        if transcription_config.get('enable_gpu', True):
            for config_key, env_var in gpu_optimization_mapping.items():
                # Store original value
                original_env[env_var] = os.environ.get(env_var)

                # Set default GPU optimization values
                default_value = "true"
                os.environ[env_var] = default_value
                logger.debug(f"🚀 Set {env_var}={default_value}")

        # Handle temporary Deepgram override (compatibility with transcription_config.py)
        method = transcription_config.get('method', 'auto')
        if method == 'deepgram':
            original_env['FORCE_DEEPGRAM_TRANSCRIPTION'] = os.environ.get('FORCE_DEEPGRAM_TRANSCRIPTION')
            os.environ['FORCE_DEEPGRAM_TRANSCRIPTION'] = "true"
            logger.info("🔄 Enabled temporary Deepgram override")
        elif 'FORCE_DEEPGRAM_TRANSCRIPTION' in os.environ:
            # Clear override if method is not deepgram
            original_env['FORCE_DEEPGRAM_TRANSCRIPTION'] = os.environ.get('FORCE_DEEPGRAM_TRANSCRIPTION')
            os.environ['FORCE_DEEPGRAM_TRANSCRIPTION'] = "false"
            logger.info("🔄 Disabled temporary Deepgram override")

        # Log final transcription configuration
        logger.info(f"🎤 Final transcription config:")
        logger.info(f"   Method: {os.environ.get('TRANSCRIPTION_METHOD', 'auto')}")
        logger.info(f"   GPU enabled: {os.environ.get('ENABLE_GPU_TRANSCRIPTION', 'true')}")
        logger.info(f"   Fallback enabled: {os.environ.get('ENABLE_FALLBACK', 'true')}")
        logger.info(f"   Cost optimization: {os.environ.get('COST_OPTIMIZATION', 'true')}")
        logger.info(f"   Deepgram override: {os.environ.get('FORCE_DEEPGRAM_TRANSCRIPTION', 'false')}")

        return original_env

    def _restore_environment(self, original_env: Dict[str, str]) -> None:
        """
        Restore original environment variables

        Args:
            original_env: Dictionary of original environment variable values
        """
        import os

        for env_var, original_value in original_env.items():
            if original_value is None:
                # Variable didn't exist originally, remove it
                if env_var in os.environ:
                    del os.environ[env_var]
            else:
                # Restore original value
                os.environ[env_var] = original_value

            logger.debug(f"Restored {env_var} to original value")


# Global pipeline wrapper instance
pipeline_wrapper = EnhancedPipelineWrapper()
