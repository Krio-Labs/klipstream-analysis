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
            ProcessingStage.DOWNLOADING: (0, 30),
            ProcessingStage.FETCHING_CHAT: (30, 35),
            ProcessingStage.TRANSCRIBING: (35, 65),
            ProcessingStage.ANALYZING: (65, 90),
            ProcessingStage.FINDING_HIGHLIGHTS: (90, 95),
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
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the integrated pipeline with enhanced progress tracking and error handling
        
        Args:
            video_url: URL of the video to process
            job_id: Unique job identifier
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary with processing results
        """
        tracker = PipelineProgressTracker(job_id, progress_callback)
        
        try:
            logger.info(f"Starting enhanced pipeline for job {job_id}")
            
            # Stage 1: Queue and initialize
            await tracker.update_progress(ProcessingStage.QUEUED, 0, "Initializing analysis...")
            
            # Import the main pipeline function
            from main import run_integrated_pipeline
            
            # Create a wrapper function for retry
            async def pipeline_execution():
                # Import and run the integrated pipeline directly to avoid thread pool issues
                # with subprocess execution in FastAPI environment
                from main import run_integrated_pipeline

                # Check if run_integrated_pipeline is async or sync
                result = run_integrated_pipeline(video_url)
                if asyncio.iscoroutine(result):
                    return await result
                else:
                    return result
            
            # Stage 2: Start downloading
            await tracker.update_progress(ProcessingStage.DOWNLOADING, 5, "Starting video download...")
            
            # Execute pipeline with retry logic
            result = await retry_processing_operation(
                pipeline_execution,
                operation_name=f"pipeline_execution_{job_id}"
            )

            # Check if pipeline completed successfully
            if result and result.get("status") == "completed":
                # Pipeline already updated status to completed, just confirm final state
                await tracker.update_progress(ProcessingStage.COMPLETED, 100, "Analysis completed successfully!")
                logger.info(f"Pipeline completed successfully for job {job_id}")
            else:
                # Pipeline failed or returned unexpected result
                error_message = result.get("error", "Unknown error") if result else "Pipeline returned no result"
                logger.error(f"Pipeline failed for job {job_id}: {error_message}")
                raise Exception(f"Pipeline execution failed: {error_message}")

            # Note: The actual pipeline (main.py) handles its own progress updates
            # We don't need to simulate progress here as it causes status conflicts
            
            logger.info(f"Pipeline completed successfully for job {job_id}")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed for job {job_id}: {str(e)}")
            
            # Classify the error
            error_info = error_classifier.classify_error(e, {
                "job_id": job_id,
                "video_url": video_url,
                "current_stage": tracker.current_stage.value,
                "stage_elapsed_time": (datetime.utcnow() - tracker.stage_start_time).total_seconds()
            })
            
            # Update progress with failure
            await tracker.update_progress(
                ProcessingStage.FAILED,
                tracker.stage_weights[tracker.current_stage][0],  # Keep current progress
                f"Failed: {error_info.error_message}"
            )
            
            # Re-raise with enhanced error info
            raise Exception(f"Pipeline failed: {error_info.error_message}") from e
    
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
            
            # Import raw pipeline
            from raw_pipeline.processor import RawPipelineProcessor
            
            processor = RawPipelineProcessor()
            
            # Create progress wrapper for raw pipeline
            async def raw_pipeline_execution():
                return await asyncio.get_event_loop().run_in_executor(
                    None, processor.process_video, video_url
                )
            
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
            
            # Import analysis pipeline
            from analysis_pipeline.processor import AnalysisPipelineProcessor
            
            processor = AnalysisPipelineProcessor()
            
            # Create progress wrapper
            async def analysis_pipeline_execution():
                return await asyncio.get_event_loop().run_in_executor(
                    None, processor.process_analysis, video_id
                )
            
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
            ProcessingStage.FETCHING_CHAT: "Downloading chat messages and metadata",
            ProcessingStage.TRANSCRIBING: "Generating transcript from audio using AI",
            ProcessingStage.ANALYZING: "Analyzing sentiment and content patterns",
            ProcessingStage.FINDING_HIGHLIGHTS: "Detecting highlights and key moments",
            ProcessingStage.COMPLETED: "Analysis completed successfully!",
            ProcessingStage.FAILED: "Analysis failed - please try again"
        }
        return descriptions.get(stage, f"Processing: {stage.value}")


# Global pipeline wrapper instance
pipeline_wrapper = EnhancedPipelineWrapper()
