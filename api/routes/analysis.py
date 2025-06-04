"""
Analysis Routes

This module contains the FastAPI routes for video analysis operations.
It provides endpoints for starting analysis jobs and managing the analysis workflow.
"""

import uuid
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..models import (
    AnalysisRequest, 
    AnalysisResponse, 
    ProcessingStage, 
    ProgressInfo,
    ErrorType,
    ErrorInfo,
    ANALYSIS_START_RESPONSE_EXAMPLE,
    ANALYSIS_ERROR_RESPONSE_EXAMPLE
)
from ..services.job_manager import JobManager, AnalysisJob
from utils.helpers import extract_video_id

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Get the global job manager instance
from ..services.job_manager import job_manager_instance as job_manager


def validate_twitch_url(url: str) -> str:
    """
    Validate and extract video ID from Twitch URL
    
    Args:
        url: Twitch VOD URL
        
    Returns:
        str: Extracted video ID
        
    Raises:
        HTTPException: If URL is invalid
    """
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")
        return video_id
    except Exception as e:
        logger.error(f"Invalid Twitch URL: {url}, Error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "status": "failed",
                "message": "Invalid Twitch URL format",
                "error": {
                    "error_type": ErrorType.INVALID_VIDEO_URL,
                    "error_code": "INVALID_URL_FORMAT",
                    "error_message": "The provided URL is not a valid Twitch VOD URL",
                    "error_details": f"URL must match pattern: https://www.twitch.tv/videos/{{video_id}}. Error: {str(e)}",
                    "is_retryable": False,
                    "suggested_action": "Please provide a valid Twitch VOD URL and try again",
                    "support_reference": f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-001"
                }
            }
        )


@router.post(
    "/analysis",
    response_model=AnalysisResponse,
    summary="Start Video Analysis",
    description="Initiate asynchronous analysis of a Twitch VOD with immediate response and background processing",
    responses={
        200: {
            "description": "Analysis started successfully",
            "content": {
                "application/json": {
                    "example": ANALYSIS_START_RESPONSE_EXAMPLE
                }
            }
        },
        400: {
            "description": "Invalid request or URL format",
            "content": {
                "application/json": {
                    "example": ANALYSIS_ERROR_RESPONSE_EXAMPLE
                }
            }
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> AnalysisResponse:
    """
    Start video analysis with immediate response
    
    This endpoint:
    1. Validates the Twitch URL
    2. Creates a unique job ID
    3. Initializes the analysis job
    4. Starts background processing
    5. Returns immediate response with job details
    
    The actual processing happens in the background, and progress can be tracked
    using the status endpoints.
    """
    try:
        # Validate URL and extract video ID
        video_id = validate_twitch_url(request.url)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        logger.info(f"Starting analysis for video {video_id} with job ID {job_id}")
        
        # Create analysis job
        analysis_job = AnalysisJob(
            id=job_id,
            video_id=video_id,
            video_url=request.url,
            status=ProcessingStage.QUEUED,
            progress_percentage=0.0,
            estimated_completion_seconds=3600,  # 1 hour default estimate
            created_at=datetime.utcnow(),
            callback_url=str(request.callback_url) if request.callback_url else None
        )
        
        # Save job to manager
        await job_manager.create_job(analysis_job)
        
        # Start background processing
        background_tasks.add_task(job_manager.process_video_analysis, analysis_job)
        
        # Create progress info
        progress = ProgressInfo(
            percentage=0.0,
            current_stage=ProcessingStage.QUEUED,
            stages_completed=0,
            total_stages=5,  # Queued, Downloading, Transcribing, Analyzing, Completed
            estimated_completion_seconds=3600,
            estimated_completion_time=datetime.utcnow() + timedelta(seconds=3600),
            stage_progress={}
        )
        
        # Return immediate response
        response = AnalysisResponse(
            status="success",
            message="Analysis started successfully",
            timestamp=datetime.utcnow(),
            job_id=job_id,
            video_id=video_id,
            progress=progress,
            metadata={
                "api_version": "2.0.0",
                "status_url": f"/api/v1/analysis/{job_id}/status",
                "stream_url": f"/api/v1/analysis/{job_id}/stream"
            }
        )
        
        logger.info(f"Analysis job {job_id} created successfully for video {video_id}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}", exc_info=True)
        
        # Create error response
        error_info = ErrorInfo(
            error_type=ErrorType.UNKNOWN_ERROR,
            error_code="ANALYSIS_START_FAILED",
            error_message="Failed to start video analysis",
            error_details=str(e),
            is_retryable=True,
            retry_after_seconds=60,
            suggested_action="Please try again in a few minutes or contact support if the issue persists",
            support_reference=f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-002"
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": "Internal server error",
                "error": error_info.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Start Video Analysis (Legacy Endpoint)",
    description="Legacy endpoint that redirects to /analysis. Maintained for backward compatibility.",
    responses={
        200: {
            "description": "Analysis started successfully",
            "content": {
                "application/json": {
                    "example": ANALYSIS_START_RESPONSE_EXAMPLE
                }
            }
        },
        400: {
            "description": "Invalid request or URL format",
            "content": {
                "application/json": {
                    "example": ANALYSIS_ERROR_RESPONSE_EXAMPLE
                }
            }
        }
    }
)
async def start_analysis_legacy(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> AnalysisResponse:
    """
    Legacy endpoint for starting video analysis

    This endpoint provides backward compatibility for clients using the old /analyze endpoint.
    It redirects to the new /analysis endpoint with the same functionality.
    """
    logger.info(f"Legacy /analyze endpoint called, redirecting to /analysis")
    return await start_analysis(request, background_tasks)


@router.get(
    "/analysis/{job_id}",
    response_model=AnalysisResponse,
    summary="Get Analysis Details",
    description="Get detailed information about an analysis job including current status and results"
)
async def get_analysis(job_id: str) -> AnalysisResponse:
    """
    Get detailed analysis information by job ID
    
    This endpoint returns comprehensive information about an analysis job,
    including current status, progress, and results (if completed).
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
                        "support_reference": f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-003"
                    }
                }
            )
        
        # Convert job to response format
        progress = ProgressInfo(
            percentage=job.progress_percentage,
            current_stage=job.status,
            stages_completed=job.stages_completed,
            total_stages=5,
            estimated_completion_seconds=job.estimated_completion_seconds,
            estimated_completion_time=job.estimated_completion_time,
            stage_progress=job.stage_progress or {}
        )
        
        response = AnalysisResponse(
            status="success" if job.status != ProcessingStage.FAILED else "failed",
            message=job.current_message or f"Analysis is {job.status.value.lower()}",
            timestamp=datetime.utcnow(),
            job_id=job.id,
            video_id=job.video_id,
            progress=progress,
            results=job.results,
            error=job.error_info,
            metadata={
                "created_at": job.created_at.isoformat(),
                "updated_at": job.last_updated.isoformat() if job.last_updated else None,
                "api_version": "2.0.0"
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": "Internal server error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
