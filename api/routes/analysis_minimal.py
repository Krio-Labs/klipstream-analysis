"""
Minimal Analysis Routes for Debugging

This is a simplified version of the analysis routes to isolate the hanging issue.
"""

import uuid
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..models import (
    AnalysisRequest, 
    AnalysisResponse, 
    ProcessingStage, 
    ProgressInfo,
    ErrorType,
    ErrorInfo
)
from utils.helpers import extract_video_id

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


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


@router.get(
    "/test",
    summary="Test Endpoint",
    description="Simple test endpoint to verify API is working"
)
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "status": "success",
        "message": "Minimal API is working",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0-minimal"
    }


@router.post(
    "/analysis-minimal",
    summary="Start Video Analysis (Minimal Version)",
    description="Minimal version of analysis endpoint for debugging"
)
async def start_analysis_minimal(request: dict):
    """
    Minimal analysis endpoint that doesn't use background tasks or Convex
    
    This endpoint:
    1. Validates the Twitch URL
    2. Creates a unique job ID
    3. Returns immediate response
    4. Does NOT start background processing (for debugging)
    """
    try:
        # Extract URL from request dict
        url = request.get('url', '')
        if not url:
            return {"status": "error", "message": "URL is required"}

        # Validate URL and extract video ID
        video_id = validate_twitch_url(url)

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        logger.info(f"Minimal analysis for video {video_id} with job ID {job_id}")
        
        # Return simple JSON response to avoid Pydantic serialization issues
        response = {
            "status": "success",
            "message": "Analysis queued (minimal version - no processing)",
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "video_id": video_id,
            "progress": {
                "percentage": 0.0,
                "current_stage": "queued",
                "stages_completed": 0,
                "total_stages": 5,
                "estimated_completion_seconds": 3600
            },
            "metadata": {
                "api_version": "2.0.0-minimal",
                "note": "This is a minimal version for debugging",
                "background_processing": False
            }
        }

        logger.info(f"Minimal analysis job {job_id} created for video {video_id}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        logger.error(f"Error in minimal analysis: {str(e)}", exc_info=True)
        
        # Create error response
        error_info = ErrorInfo(
            error_type=ErrorType.UNKNOWN_ERROR,
            error_code="MINIMAL_ANALYSIS_FAILED",
            error_message="Failed to start minimal video analysis",
            error_details=str(e),
            is_retryable=True,
            retry_after_seconds=60,
            suggested_action="Please try again or contact support",
            support_reference=f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-MIN"
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": "Internal server error",
                "error": error_info.model_dump(),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get(
    "/analysis-minimal/{job_id}",
    response_model=AnalysisResponse,
    summary="Get Minimal Analysis Status",
    description="Get status for minimal analysis job (always returns queued)"
)
async def get_analysis_minimal(job_id: str) -> AnalysisResponse:
    """
    Get minimal analysis status (for debugging)
    """
    try:
        # Create mock progress info
        progress = ProgressInfo(
            percentage=0.0,
            current_stage=ProcessingStage.QUEUED,
            stages_completed=0,
            total_stages=5,
            estimated_completion_seconds=3600,
            estimated_completion_time=datetime.utcnow() + timedelta(seconds=3600),
            stage_progress={}
        )
        
        response = AnalysisResponse(
            status="success",
            message="Minimal analysis job (no actual processing)",
            timestamp=datetime.utcnow(),
            job_id=job_id,
            video_id="minimal-test",
            progress=progress,
            metadata={
                "api_version": "2.0.0-minimal",
                "note": "This is a minimal version for debugging"
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting minimal analysis {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": "Internal server error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
