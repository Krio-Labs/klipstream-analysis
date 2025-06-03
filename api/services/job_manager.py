"""
Job Manager Service

This module handles the creation, tracking, and processing of analysis jobs.
It manages the background processing of video analysis tasks and maintains job state.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from ..models import (
    ProcessingStage, 
    ErrorType, 
    ErrorInfo, 
    AnalysisResults
)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class AnalysisJob:
    """
    Data class representing an analysis job
    """
    id: str
    video_id: str
    video_url: str
    status: ProcessingStage
    progress_percentage: float
    estimated_completion_seconds: int
    created_at: datetime
    callback_url: Optional[str] = None

    # Progress tracking
    current_message: Optional[str] = None
    last_updated: Optional[datetime] = None
    stages_completed: int = 0
    total_stages: int = 5  # Total number of processing stages
    stage_progress: Optional[Dict[str, float]] = field(default_factory=dict)
    estimated_completion_time: Optional[datetime] = None

    # Results and errors
    results: Optional[AnalysisResults] = None
    error_info: Optional[ErrorInfo] = None

    # Processing metadata
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    retry_count: int = 0
    
    def update_progress(self, stage: ProcessingStage, percentage: float, message: str = None):
        """Update job progress"""
        self.status = stage
        self.progress_percentage = percentage
        self.current_message = message or f"Processing: {stage.value}"
        self.last_updated = datetime.utcnow()
        
        # Update estimated completion time
        if percentage > 0 and stage != ProcessingStage.COMPLETED:
            elapsed_seconds = (datetime.utcnow() - self.created_at).total_seconds()
            estimated_total = elapsed_seconds / (percentage / 100)
            self.estimated_completion_seconds = int(estimated_total - elapsed_seconds)
            self.estimated_completion_time = datetime.utcnow() + timedelta(seconds=self.estimated_completion_seconds)
        
        # Update stages completed
        stage_order = [
            ProcessingStage.QUEUED,
            ProcessingStage.DOWNLOADING,
            ProcessingStage.TRANSCRIBING,
            ProcessingStage.ANALYZING,
            ProcessingStage.COMPLETED
        ]
        
        if stage in stage_order:
            self.stages_completed = stage_order.index(stage)
    
    def mark_failed(self, error_info: ErrorInfo):
        """Mark job as failed with error information"""
        self.status = ProcessingStage.FAILED
        self.error_info = error_info
        self.last_updated = datetime.utcnow()
        self.processing_completed_at = datetime.utcnow()
    
    def mark_completed(self, results: AnalysisResults):
        """Mark job as completed with results"""
        self.status = ProcessingStage.COMPLETED
        self.progress_percentage = 100.0
        self.results = results
        self.last_updated = datetime.utcnow()
        self.processing_completed_at = datetime.utcnow()
        self.estimated_completion_seconds = 0
        self.stages_completed = self.total_stages


class JobManager:
    """
    Manager for analysis jobs
    
    This class handles the creation, tracking, and processing of analysis jobs.
    It maintains an in-memory store of jobs and coordinates background processing.
    """
    
    def __init__(self):
        self.jobs: Dict[str, AnalysisJob] = {}
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        logger.info("JobManager initialized")
    
    async def create_job(self, job: AnalysisJob) -> None:
        """
        Create and store a new analysis job
        
        Args:
            job: The AnalysisJob to create
        """
        self.jobs[job.id] = job
        logger.info(f"Created job {job.id} for video {job.video_id}")
    
    async def get_job(self, job_id: str) -> Optional[AnalysisJob]:
        """
        Get a job by its ID
        
        Args:
            job_id: The job ID to look up
            
        Returns:
            The AnalysisJob if found, None otherwise
        """
        return self.jobs.get(job_id)
    
    async def list_jobs(self) -> List[AnalysisJob]:
        """
        Get a list of all jobs
        
        Returns:
            List of all AnalysisJob objects
        """
        return list(self.jobs.values())
    
    async def update_job_progress(
        self, 
        job_id: str, 
        stage: ProcessingStage, 
        percentage: float, 
        message: str = None
    ) -> bool:
        """
        Update the progress of a job
        
        Args:
            job_id: The job ID to update
            stage: The current processing stage
            percentage: Progress percentage (0-100)
            message: Optional status message
            
        Returns:
            True if update was successful, False if job not found
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Attempted to update non-existent job {job_id}")
            return False
        
        previous_stage = job.status
        job.update_progress(stage, percentage, message)
        logger.info(f"Updated job {job_id}: {stage.value} ({percentage:.1f}%)")

        # Send webhook notifications
        await self._send_webhook_notifications(job, previous_stage)

        # Call any registered progress callbacks
        if job_id in self.progress_callbacks:
            for callback in self.progress_callbacks[job_id]:
                try:
                    await callback(job)
                except Exception as e:
                    logger.error(f"Error in progress callback for job {job_id}: {str(e)}")

        return True
    
    async def register_progress_callback(self, job_id: str, callback: Callable) -> None:
        """
        Register a callback function to be called when job progress updates
        
        Args:
            job_id: The job ID to monitor
            callback: Async function to call on progress updates
        """
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)

    async def _send_webhook_notifications(self, job: AnalysisJob, previous_stage: ProcessingStage):
        """Send webhook notifications for job updates"""
        try:
            from .webhook_manager import webhook_manager, WebhookEvent

            # Determine which events to send
            events_to_send = []

            # Job started event
            if previous_stage == ProcessingStage.QUEUED and job.status != ProcessingStage.QUEUED:
                events_to_send.append(WebhookEvent.JOB_STARTED)

            # Stage changed event
            if previous_stage != job.status:
                events_to_send.append(WebhookEvent.JOB_STAGE_CHANGED)

            # Progress event (send every 10% progress or stage change)
            if (job.progress_percentage % 10 == 0 or
                previous_stage != job.status or
                job.status in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]):
                events_to_send.append(WebhookEvent.JOB_PROGRESS)

            # Completion events
            if job.status == ProcessingStage.COMPLETED:
                events_to_send.append(WebhookEvent.JOB_COMPLETED)
            elif job.status == ProcessingStage.FAILED:
                events_to_send.append(WebhookEvent.JOB_FAILED)

            # Send webhook notifications
            for event in events_to_send:
                payload = {
                    "job_id": job.id,
                    "video_id": job.video_id,
                    "status": job.status.value,
                    "progress": {
                        "percentage": job.progress_percentage,
                        "current_stage": job.status.value,
                        "stages_completed": job.stages_completed,
                        "total_stages": job.total_stages,
                        "estimated_completion_seconds": job.estimated_completion_seconds,
                        "message": job.current_message
                    },
                    "created_at": job.created_at.isoformat(),
                    "updated_at": job.last_updated.isoformat()
                }

                # Add results if completed
                if job.status == ProcessingStage.COMPLETED and job.results:
                    payload["results"] = {
                        "video_file_url": job.results.video_file_url,
                        "transcript_file_url": job.results.transcript_file_url,
                        "highlights_file_url": job.results.highlights_file_url,
                        "analysis_report_url": job.results.analysis_report_url,
                        "video_duration_seconds": job.results.video_duration_seconds,
                        "processing_time_seconds": job.results.processing_time_seconds
                    }

                # Add error info if failed
                if job.status == ProcessingStage.FAILED and job.error_info:
                    payload["error"] = {
                        "error_type": job.error_info.error_type.value,
                        "error_code": job.error_info.error_code,
                        "error_message": job.error_info.error_message,
                        "is_retryable": job.error_info.is_retryable,
                        "suggested_action": job.error_info.suggested_action,
                        "support_reference": job.error_info.support_reference
                    }

                await webhook_manager.send_webhook(
                    event=event,
                    job_id=job.id,
                    video_id=job.video_id,
                    payload=payload
                )

        except Exception as e:
            logger.error(f"Error sending webhook notifications for job {job.id}: {str(e)}")

    async def process_video_analysis(self, job: AnalysisJob) -> None:
        """
        Process video analysis in the background
        
        This method coordinates the entire analysis pipeline and updates job progress.
        It integrates with the existing pipeline functions while providing progress tracking.
        
        Args:
            job: The AnalysisJob to process
        """
        logger.info(f"Starting background processing for job {job.id}")
        
        try:
            # Import here to avoid circular imports
            from utils.convex_client_updated import ConvexManager
            
            # Initialize Convex manager
            convex_manager = ConvexManager()
            
            # Mark processing as started
            job.processing_started_at = datetime.utcnow()
            await self.update_job_progress(job.id, ProcessingStage.QUEUED, 0.0, "Analysis queued")
            
            # Update Convex status
            convex_manager.update_video_status(job.video_id, "Queued")
            
            # Create a progress callback for the pipeline
            async def pipeline_progress_callback(stage: str, percentage: float, message: str = None):
                """Callback to update job progress from pipeline"""
                # Map pipeline stages to our enum
                stage_mapping = {
                    "downloading": ProcessingStage.DOWNLOADING,
                    "fetching_chat": ProcessingStage.FETCHING_CHAT,
                    "transcribing": ProcessingStage.TRANSCRIBING,
                    "analyzing": ProcessingStage.ANALYZING,
                    "finding_highlights": ProcessingStage.FINDING_HIGHLIGHTS,
                    "completed": ProcessingStage.COMPLETED
                }

                mapped_stage = stage_mapping.get(stage.lower(), ProcessingStage.ANALYZING)
                await self.update_job_progress(job.id, mapped_stage, percentage, message)

                # Update Convex status
                convex_status_mapping = {
                    ProcessingStage.DOWNLOADING: "Downloading",
                    ProcessingStage.FETCHING_CHAT: "Fetching chat",
                    ProcessingStage.TRANSCRIBING: "Transcribing",
                    ProcessingStage.ANALYZING: "Analyzing",
                    ProcessingStage.FINDING_HIGHLIGHTS: "Finding highlights",
                    ProcessingStage.COMPLETED: "Completed"
                }
                convex_status = convex_status_mapping.get(mapped_stage, "Analyzing")
                convex_manager.update_video_status(job.video_id, convex_status)

                # Also update Convex with job progress data
                progress_data = {
                    "progressPercentage": percentage,
                    "currentStage": convex_status,
                    "estimatedCompletionSeconds": job.estimated_completion_seconds,
                    "updatedAt": int(datetime.utcnow().timestamp() * 1000)  # Convex expects milliseconds
                }
                convex_manager.update_job_progress(job.video_id, job.id, progress_data)

            # Run the integrated pipeline with enhanced progress tracking
            from .pipeline_wrapper import pipeline_wrapper

            result = await pipeline_wrapper.run_integrated_pipeline_with_tracking(
                job.video_url,
                job.id,
                pipeline_progress_callback
            )
            
            if result and result.get("status") == "completed":
                # Create results object
                analysis_results = AnalysisResults(
                    video_file_url=result.get("files", {}).get("video_file", ""),
                    transcript_file_url=result.get("files", {}).get("transcript_files", {}).get("json", ""),
                    highlights_file_url=result.get("files", {}).get("analysis_files", {}).get("highlights", ""),
                    analysis_report_url=result.get("files", {}).get("analysis_files", {}).get("integrated", ""),
                    video_duration_seconds=result.get("video_duration", 0.0),
                    transcript_word_count=result.get("transcript_word_count", 0),
                    highlights_count=result.get("highlights_count", 0),
                    sentiment_score=result.get("sentiment_score", 0.0),
                    processing_time_seconds=result.get("total_duration", 0.0),
                    file_sizes=result.get("file_sizes", {})
                )
                
                # Mark job as completed
                job.mark_completed(analysis_results)
                logger.info(f"Job {job.id} completed successfully")
                
                # Update Convex status
                convex_manager.update_video_status(job.video_id, "Completed")
                
            else:
                # Pipeline failed
                error_message = result.get("error", "Unknown error") if result else "Pipeline returned no result"
                error_info = ErrorInfo(
                    error_type=ErrorType.UNKNOWN_ERROR,
                    error_code="PIPELINE_FAILED",
                    error_message="Video analysis pipeline failed",
                    error_details=error_message,
                    is_retryable=True,
                    retry_after_seconds=300,
                    suggested_action="Please try again or contact support if the issue persists",
                    support_reference=f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{job.id[:8]}"
                )
                
                job.mark_failed(error_info)
                logger.error(f"Job {job.id} failed: {error_message}")
                
                # Update Convex status
                convex_manager.update_video_status(job.video_id, "Failed")
                
        except Exception as e:
            logger.error(f"Error processing job {job.id}: {str(e)}", exc_info=True)

            # Use enhanced error classification
            from .error_handler import error_classifier
            error_info = error_classifier.classify_error(e, {
                "job_id": job.id,
                "video_id": job.video_id,
                "stage": "pipeline_execution",
                "processing_time": (datetime.utcnow() - job.created_at).total_seconds()
            })

            job.mark_failed(error_info)
            
            # Update Convex status
            try:
                from utils.convex_client_updated import ConvexManager
                convex_manager = ConvexManager()
                convex_manager.update_video_status(job.video_id, "Failed")
            except Exception as convex_error:
                logger.error(f"Failed to update Convex status: {str(convex_error)}")
        
        finally:
            # Clean up progress callbacks
            if job.id in self.progress_callbacks:
                del self.progress_callbacks[job.id]
            
            logger.info(f"Background processing completed for job {job.id}")


# Global job manager instance
job_manager_instance = JobManager()
