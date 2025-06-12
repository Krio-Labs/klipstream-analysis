"""
Status Routes

This module contains the FastAPI routes for status tracking and monitoring.
It provides endpoints for checking job status and streaming real-time updates.
"""

import json
import asyncio
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from ..models import JobStatus, ProcessingStage, ProgressInfo, ErrorType
from ..services.job_manager import JobManager

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Get the global job manager instance
from ..services.job_manager import job_manager_instance as job_manager


@router.get(
    "/analysis/{job_id}/status",
    response_model=JobStatus,
    summary="Get Job Status",
    description="Get the current status of an analysis job"
)
async def get_analysis_status(job_id: str) -> JobStatus:
    """
    Get current status of an analysis job
    
    This endpoint returns the current status, progress, and other details
    of an analysis job identified by its job ID.
    """
    try:
        # Get job from manager
        job = await job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "failed",
                    "message": "Analysis job not found",
                    "error": {
                        "error_type": ErrorType.UNKNOWN_ERROR,
                        "error_code": "JOB_NOT_FOUND",
                        "error_message": f"No analysis job found with ID: {job_id}",
                        "error_details": "The job may have expired or the ID is incorrect",
                        "is_retryable": False,
                        "suggested_action": "Please check the job ID and try again",
                        "support_reference": f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-004"
                    }
                }
            )
        
        # Create progress info
        progress = ProgressInfo(
            percentage=job.progress_percentage,
            current_stage=job.status,
            stages_completed=job.stages_completed,
            total_stages=5,
            estimated_completion_seconds=job.estimated_completion_seconds,
            estimated_completion_time=job.estimated_completion_time,
            stage_progress=job.stage_progress or {}
        )
        
        # Create status response
        status_response = JobStatus(
            job_id=job.id,
            video_id=job.video_id,
            status=job.status,
            progress=progress,
            created_at=job.created_at,
            updated_at=job.last_updated or job.created_at,
            error=job.error_info
        )
        
        return status_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": "Internal server error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get(
    "/analysis/{job_id}/stream",
    summary="Stream Job Status",
    description="Stream real-time status updates for an analysis job using Server-Sent Events",
    responses={
        200: {
            "description": "Server-Sent Events stream",
            "content": {
                "text/plain": {
                    "example": "data: {\"job_id\":\"123\",\"status\":\"Downloading\",\"progress\":25.5}\n\n"
                }
            }
        },
        404: {
            "description": "Job not found"
        }
    }
)
async def stream_analysis_status(job_id: str):
    """
    Stream real-time analysis status updates using Server-Sent Events (SSE)

    This endpoint provides a continuous stream of status updates for an analysis job.
    The client can listen to this stream to get real-time progress updates.

    Features:
    - Real-time progress updates every 2 seconds
    - Automatic stream closure when job completes/fails
    - Detailed error information
    - Stage-specific progress tracking
    - Connection health monitoring

    The stream will automatically close when the job completes or fails.
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        """
        Generate Server-Sent Events for job status updates with enhanced features
        """
        connection_id = f"conn_{job_id}_{int(datetime.utcnow().timestamp())}"
        logger.info(f"Starting SSE stream {connection_id} for job {job_id}")

        try:
            # Check if job exists initially
            job = await job_manager.get_job(job_id)
            if not job:
                error_data = {
                    "event": "error",
                    "status": "error",
                    "message": "Job not found",
                    "job_id": job_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "connection_id": connection_id,
                    "error": {
                        "error_type": ErrorType.UNKNOWN_ERROR.value,
                        "error_code": "JOB_NOT_FOUND",
                        "error_message": f"No analysis job found with ID: {job_id}",
                        "is_retryable": False,
                        "suggested_action": "Please check the job ID and try again"
                    }
                }
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                return

            # Send initial connection confirmation
            initial_data = {
                "event": "connected",
                "message": f"Connected to job {job_id} status stream",
                "job_id": job_id,
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"event: connected\ndata: {json.dumps(initial_data)}\n\n"

            # Track last sent data to avoid duplicate updates
            last_progress = -1
            last_stage = None
            update_count = 0

            # Stream status updates until job completes or fails
            while True:
                try:
                    # Get current job status
                    current_job = await job_manager.get_job(job_id)

                    if not current_job:
                        # Job disappeared - send error and close stream
                        error_data = {
                            "event": "error",
                            "status": "error",
                            "message": "Job no longer exists",
                            "job_id": job_id,
                            "connection_id": connection_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                        break

                    # Check if there's a meaningful update
                    progress_changed = abs(current_job.progress_percentage - last_progress) >= 0.1
                    stage_changed = current_job.status != last_stage

                    if progress_changed or stage_changed or update_count % 10 == 0:  # Send heartbeat every 20 seconds
                        # Prepare detailed status data
                        status_data = {
                            "event": "progress",
                            "job_id": current_job.id,
                            "video_id": current_job.video_id,
                            "status": current_job.status.value,
                            "progress": {
                                "percentage": round(current_job.progress_percentage, 2),
                                "current_stage": current_job.status.value,
                                "stages_completed": current_job.stages_completed,
                                "total_stages": 5,
                                "estimated_completion_seconds": current_job.estimated_completion_seconds,
                                "stage_progress": current_job.stage_progress or {}
                            },
                            "message": current_job.current_message or f"Processing: {current_job.status.value}",
                            "timestamp": datetime.utcnow().isoformat(),
                            "connection_id": connection_id,
                            "update_count": update_count,
                            "processing_time_seconds": (
                                (datetime.utcnow() - current_job.created_at).total_seconds()
                                if current_job.created_at else 0
                            )
                        }

                        # Add timing information
                        if current_job.processing_started_at:
                            status_data["processing_duration"] = (
                                datetime.utcnow() - current_job.processing_started_at
                            ).total_seconds()

                        # Add error details if job failed
                        if current_job.status == ProcessingStage.FAILED and current_job.error_info:
                            status_data["event"] = "failed"
                            status_data["error"] = {
                                "error_type": current_job.error_info.error_type.value,
                                "error_code": current_job.error_info.error_code,
                                "error_message": current_job.error_info.error_message,
                                "error_details": current_job.error_info.error_details,
                                "is_retryable": current_job.error_info.is_retryable,
                                "suggested_action": current_job.error_info.suggested_action,
                                "support_reference": current_job.error_info.support_reference
                            }

                        # Add enhanced results if job completed
                        if current_job.status == ProcessingStage.COMPLETED:
                            status_data["event"] = "completed"

                            # Enhanced results with all URL fields from main.py pipeline
                            results = {
                                # Core file URLs (updated to match main.py output)
                                "video_url": f"gs://klipstream-vods-raw/{current_job.video_id}/video.mp4",
                                "audio_url": f"gs://klipstream-vods-raw/{current_job.video_id}/audio.mp3",
                                "waveform_url": f"gs://klipstream-vods-raw/{current_job.video_id}/waveform.json",
                                "transcript_url": f"gs://klipstream-transcripts/{current_job.video_id}/segments.csv",
                                "transcriptWords_url": f"gs://klipstream-transcripts/{current_job.video_id}/words.csv",
                                "chat_url": f"gs://klipstream-chatlogs/{current_job.video_id}/chat.csv",
                                "analysis_url": f"gs://klipstream-analysis/{current_job.video_id}/audio/audio_{current_job.video_id}_sentiment.csv",

                                # Additional analysis URLs
                                "highlights_url": f"gs://klipstream-analysis/{current_job.video_id}/audio/audio_{current_job.video_id}_highlights.csv",
                                "audio_sentiment_url": f"gs://klipstream-analysis/{current_job.video_id}/audio/audio_{current_job.video_id}_sentiment.csv",
                                "chat_sentiment_url": f"gs://klipstream-analysis/{current_job.video_id}/chat/{current_job.video_id}_chat_sentiment.csv",
                                "integrated_analysis_url": f"gs://klipstream-analysis/{current_job.video_id}/integrated_{current_job.video_id}.json"
                            }

                            # Add job results if available
                            if current_job.results:
                                results.update({
                                    "video_duration_seconds": current_job.results.video_duration_seconds,
                                    "transcript_word_count": current_job.results.transcript_word_count,
                                    "highlights_count": current_job.results.highlights_count,
                                    "sentiment_score": current_job.results.sentiment_score,
                                    "processing_time_seconds": current_job.results.processing_time_seconds,
                                    "file_sizes": current_job.results.file_sizes
                                })

                            status_data["results"] = results

                            # Add pipeline metadata if available
                            if hasattr(current_job, 'pipeline_metadata'):
                                status_data["pipeline_metadata"] = current_job.pipeline_metadata

                        # Send the update
                        event_type = status_data["event"]
                        yield f"event: {event_type}\ndata: {json.dumps(status_data)}\n\n"

                        # Update tracking variables
                        last_progress = current_job.progress_percentage
                        last_stage = current_job.status

                    # Check if job is finished
                    if current_job.status in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]:
                        logger.info(f"Job {job_id} finished with status: {current_job.status.value}")

                        # Send final completion event
                        final_data = {
                            "event": "stream_ended",
                            "message": f"Stream ended - job {current_job.status.value.lower()}",
                            "job_id": job_id,
                            "final_status": current_job.status.value,
                            "connection_id": connection_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "total_updates": update_count
                        }
                        yield f"event: stream_ended\ndata: {json.dumps(final_data)}\n\n"
                        break

                    # Wait before next update (2 seconds for more responsive updates)
                    await asyncio.sleep(2)
                    update_count += 1

                except Exception as e:
                    logger.error(f"Error in status stream {connection_id}: {str(e)}")
                    error_data = {
                        "event": "error",
                        "status": "error",
                        "message": "Stream error occurred",
                        "job_id": job_id,
                        "connection_id": connection_id,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                        "is_retryable": True,
                        "suggested_action": "Please refresh the connection"
                    }
                    yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                    break

        except Exception as e:
            logger.error(f"Fatal error in status stream {connection_id}: {str(e)}")
            error_data = {
                "event": "fatal_error",
                "status": "error",
                "message": "Fatal stream error",
                "job_id": job_id,
                "connection_id": connection_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "is_retryable": False,
                "suggested_action": "Please contact support if this issue persists"
            }
            yield f"event: fatal_error\ndata: {json.dumps(error_data)}\n\n"

        finally:
            logger.info(f"SSE stream {connection_id} closed for job {job_id}")

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Content-Type": "text/plain; charset=utf-8"
        }
    )


@router.get(
    "/jobs",
    summary="List All Jobs",
    description="Get a list of all analysis jobs (for debugging and monitoring)"
)
async def list_jobs():
    """
    List all analysis jobs
    
    This endpoint returns a list of all analysis jobs in the system.
    Useful for debugging and monitoring purposes.
    """
    try:
        jobs = await job_manager.list_jobs()
        
        job_summaries = []
        for job in jobs:
            job_summaries.append({
                "job_id": job.id,
                "video_id": job.video_id,
                "status": job.status.value,
                "progress": job.progress_percentage,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.last_updated.isoformat() if job.last_updated else None
            })
        
        return {
            "status": "success",
            "message": f"Found {len(job_summaries)} jobs",
            "jobs": job_summaries,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": "Internal server error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
