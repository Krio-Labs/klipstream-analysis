"""
Status Manager Service

This module handles status tracking and updates for analysis jobs.
It provides utilities for managing job status and progress information.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from ..models import ProcessingStage, StatusUpdate
from .job_manager import AnalysisJob

# Set up logging
logger = logging.getLogger(__name__)


class StatusManager:
    """
    Manager for job status tracking and updates
    
    This class provides utilities for tracking job status, creating status updates,
    and managing status-related operations.
    """
    
    def __init__(self):
        self.status_history: Dict[str, list] = {}
        logger.info("StatusManager initialized")
    
    def create_status_update(self, job: AnalysisJob) -> StatusUpdate:
        """
        Create a status update object from a job
        
        Args:
            job: The AnalysisJob to create status update for
            
        Returns:
            StatusUpdate object with current job information
        """
        return StatusUpdate(
            video_id=job.video_id,
            stage=job.status,
            progress_percentage=job.progress_percentage,
            estimated_completion_seconds=job.estimated_completion_seconds or 0,
            message=job.current_message or f"Processing: {job.status.value}",
            timestamp=datetime.utcnow().isoformat(),
            error_details=job.error_info.error_details if job.error_info else None
        )
    
    def record_status_change(self, job_id: str, old_status: ProcessingStage, new_status: ProcessingStage):
        """
        Record a status change for tracking purposes
        
        Args:
            job_id: The job ID
            old_status: Previous status
            new_status: New status
        """
        if job_id not in self.status_history:
            self.status_history[job_id] = []
        
        self.status_history[job_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "old_status": old_status.value if old_status else None,
            "new_status": new_status.value,
            "transition": f"{old_status.value if old_status else 'None'} -> {new_status.value}"
        })
        
        logger.info(f"Status change recorded for job {job_id}: {old_status} -> {new_status}")
    
    def get_status_history(self, job_id: str) -> Optional[list]:
        """
        Get the status history for a job
        
        Args:
            job_id: The job ID
            
        Returns:
            List of status changes or None if no history exists
        """
        return self.status_history.get(job_id)
    
    def calculate_stage_progress(self, stage: ProcessingStage, sub_progress: float = 0.0) -> float:
        """
        Calculate overall progress percentage based on current stage and sub-progress
        
        Args:
            stage: Current processing stage
            sub_progress: Progress within the current stage (0-100)
            
        Returns:
            Overall progress percentage (0-100)
        """
        # Define stage weights (how much each stage contributes to overall progress)
        stage_weights = {
            ProcessingStage.QUEUED: (0, 0),      # 0% - 0%
            ProcessingStage.DOWNLOADING: (0, 30),  # 0% - 30%
            ProcessingStage.FETCHING_CHAT: (30, 35),  # 30% - 35%
            ProcessingStage.TRANSCRIBING: (35, 55),   # 35% - 55%
            ProcessingStage.ANALYZING: (55, 85),      # 55% - 85%
            ProcessingStage.FINDING_HIGHLIGHTS: (85, 95),  # 85% - 95%
            ProcessingStage.COMPLETED: (100, 100),   # 100%
            ProcessingStage.FAILED: (0, 0)       # 0% (failed)
        }
        
        if stage not in stage_weights:
            return 0.0
        
        start_percent, end_percent = stage_weights[stage]
        
        if stage == ProcessingStage.COMPLETED:
            return 100.0
        elif stage == ProcessingStage.FAILED:
            return 0.0
        else:
            # Calculate progress within the stage range
            stage_range = end_percent - start_percent
            stage_contribution = (sub_progress / 100.0) * stage_range
            return start_percent + stage_contribution
    
    def estimate_completion_time(self, job: AnalysisJob) -> int:
        """
        Estimate completion time based on current progress and elapsed time
        
        Args:
            job: The AnalysisJob to estimate for
            
        Returns:
            Estimated seconds until completion
        """
        if job.status == ProcessingStage.COMPLETED:
            return 0
        
        if job.status == ProcessingStage.FAILED:
            return 0
        
        if job.progress_percentage <= 0:
            return 3600  # Default 1 hour estimate
        
        # Calculate elapsed time
        elapsed_seconds = (datetime.utcnow() - job.created_at).total_seconds()
        
        # Estimate total time based on current progress
        estimated_total_seconds = elapsed_seconds / (job.progress_percentage / 100.0)
        
        # Calculate remaining time
        remaining_seconds = estimated_total_seconds - elapsed_seconds
        
        # Ensure minimum and maximum bounds
        remaining_seconds = max(0, min(remaining_seconds, 7200))  # Max 2 hours
        
        return int(remaining_seconds)
    
    def get_stage_description(self, stage: ProcessingStage) -> str:
        """
        Get a human-readable description of a processing stage
        
        Args:
            stage: The processing stage
            
        Returns:
            Human-readable description
        """
        descriptions = {
            ProcessingStage.QUEUED: "Your video has been queued for processing",
            ProcessingStage.DOWNLOADING: "Downloading video and extracting audio",
            ProcessingStage.FETCHING_CHAT: "Downloading chat messages and metadata",
            ProcessingStage.TRANSCRIBING: "Generating transcript from audio",
            ProcessingStage.ANALYZING: "Analyzing sentiment and content",
            ProcessingStage.FINDING_HIGHLIGHTS: "Detecting highlights and key moments",
            ProcessingStage.COMPLETED: "Analysis completed successfully",
            ProcessingStage.FAILED: "Analysis failed - please try again"
        }
        
        return descriptions.get(stage, f"Processing: {stage.value}")
    
    def cleanup_old_history(self, max_age_hours: int = 24):
        """
        Clean up old status history entries
        
        Args:
            max_age_hours: Maximum age of history entries to keep (in hours)
        """
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        jobs_to_remove = []
        for job_id, history in self.status_history.items():
            # Filter out old entries
            filtered_history = [
                entry for entry in history
                if datetime.fromisoformat(entry["timestamp"]).timestamp() > cutoff_time
            ]
            
            if filtered_history:
                self.status_history[job_id] = filtered_history
            else:
                jobs_to_remove.append(job_id)
        
        # Remove jobs with no recent history
        for job_id in jobs_to_remove:
            del self.status_history[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up status history for {len(jobs_to_remove)} old jobs")


# Global status manager instance
status_manager_instance = StatusManager()
