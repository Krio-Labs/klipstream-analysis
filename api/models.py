"""
Pydantic Models for API Request/Response Validation

This module defines all the data models used by the KlipStream Analysis API
for request validation, response formatting, and internal data structures.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class ProcessingStage(str, Enum):
    """Enumeration of processing stages"""
    QUEUED = "Queued"
    DOWNLOADING = "Downloading"
    GENERATING_WAVEFORM = "Generating waveform"
    TRANSCRIBING = "Transcribing"
    FETCHING_CHAT = "Fetching chat"
    ANALYZING = "Analyzing"
    FINDING_HIGHLIGHTS = "Finding highlights"
    COMPLETED = "Completed"
    FAILED = "Failed"


class ErrorType(str, Enum):
    """Enumeration of error types"""
    NETWORK_ERROR = "network_error"
    INVALID_VIDEO_URL = "invalid_video_url"
    VIDEO_TOO_LARGE = "video_too_large"
    PROCESSING_TIMEOUT = "processing_timeout"
    INSUFFICIENT_RESOURCES = "insufficient_resources"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    UNKNOWN_ERROR = "unknown_error"


class TranscriptionMethod(str, Enum):
    """Enumeration of available transcription methods"""
    AUTO = "auto"
    PARAKEET = "parakeet"
    DEEPGRAM = "deepgram"
    HYBRID = "hybrid"


class TranscriptionConfig(BaseModel):
    """Configuration for transcription processing"""
    method: TranscriptionMethod = Field(TranscriptionMethod.AUTO, description="Transcription method to use")
    enable_gpu: bool = Field(True, description="Enable GPU acceleration when available")
    enable_fallback: bool = Field(True, description="Enable fallback to alternative methods on failure")
    cost_optimization: bool = Field(True, description="Enable cost optimization for method selection")

    class Config:
        json_schema_extra = {
            "example": {
                "method": "auto",
                "enable_gpu": True,
                "enable_fallback": True,
                "cost_optimization": True
            }
        }


class AnalysisRequest(BaseModel):
    """Request model for starting video analysis"""
    url: str = Field(..., description="Twitch VOD URL to analyze")
    callback_url: Optional[HttpUrl] = Field(None, description="Optional webhook URL for status updates")
    transcription_config: Optional[TranscriptionConfig] = Field(None, description="Optional transcription configuration")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.twitch.tv/videos/2434635255",
                "callback_url": "https://your-app.com/api/webhook/video-status",
                "transcription_config": {
                    "method": "auto",
                    "enable_gpu": True,
                    "cost_optimization": True
                }
            }
        }


class ProgressInfo(BaseModel):
    """Progress information for ongoing analysis"""
    percentage: float = Field(..., ge=0, le=100, description="Progress percentage (0-100)")
    current_stage: ProcessingStage = Field(..., description="Current processing stage")
    stages_completed: int = Field(..., ge=0, description="Number of stages completed")
    total_stages: int = Field(..., gt=0, description="Total number of stages")
    estimated_completion_seconds: Optional[int] = Field(None, description="Estimated seconds until completion")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion timestamp")
    stage_progress: Dict[str, float] = Field(default_factory=dict, description="Progress for each stage")


class ErrorInfo(BaseModel):
    """Error information when processing fails"""
    error_type: ErrorType = Field(..., description="Type of error that occurred")
    error_code: str = Field(..., description="Specific error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_details: str = Field(..., description="Detailed error information")
    is_retryable: bool = Field(..., description="Whether the operation can be retried")
    retry_after_seconds: Optional[int] = Field(None, description="Recommended retry delay in seconds")
    suggested_action: str = Field(..., description="Suggested action for the user")
    support_reference: str = Field(..., description="Reference ID for support")


class AnalysisResults(BaseModel):
    """Analysis results when processing is complete"""
    # Core file URLs (using exact Convex field names)
    video_url: Optional[str] = Field(None, description="URL to the processed video file")
    audio_url: Optional[str] = Field(None, description="URL to the processed audio file")
    waveform_url: Optional[str] = Field(None, description="URL to the waveform visualization file")
    transcript_url: Optional[str] = Field(None, description="URL to the transcript segments file")
    transcriptWords_url: Optional[str] = Field(None, description="URL to the transcript words file")
    chat_url: Optional[str] = Field(None, description="URL to the chat log file")
    analysis_url: Optional[str] = Field(None, description="URL to the integrated analysis file")

    # Additional file URLs
    highlights_url: Optional[str] = Field(None, description="URL to the highlights file")
    audio_sentiment_url: Optional[str] = Field(None, description="URL to the audio sentiment analysis file")
    chat_sentiment_url: Optional[str] = Field(None, description="URL to the chat sentiment analysis file")

    # Summary statistics
    video_duration_seconds: Optional[float] = Field(None, description="Video duration in seconds")
    transcript_word_count: Optional[int] = Field(None, description="Number of words in transcript")
    highlights_count: Optional[int] = Field(None, description="Number of highlights detected")
    sentiment_score: Optional[float] = Field(None, description="Overall sentiment score")

    # Processing statistics
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time")
    file_sizes: Optional[Dict[str, int]] = Field(None, description="File sizes in bytes")

    # Transcription metadata
    transcription_method_used: Optional[str] = Field(None, description="Transcription method that was used")
    transcription_cost_estimate: Optional[float] = Field(None, description="Estimated transcription cost in USD")
    gpu_used: Optional[bool] = Field(None, description="Whether GPU acceleration was used")


class AnalysisResponse(BaseModel):
    """Standardized response format for all analysis endpoints"""
    # Status information
    status: str = Field(..., description="Response status (success, processing, failed)")
    message: str = Field(..., description="Human-readable status message")
    timestamp: datetime = Field(..., description="Response timestamp")
    
    # Job information
    job_id: str = Field(..., description="Unique job identifier")
    video_id: str = Field(..., description="Twitch video ID")
    
    # Progress tracking
    progress: ProgressInfo = Field(..., description="Current progress information")
    
    # Results (when completed)
    results: Optional[AnalysisResults] = Field(None, description="Analysis results (when completed)")
    
    # Error information (when failed)
    error: Optional[ErrorInfo] = Field(None, description="Error information (when failed)")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StatusUpdate(BaseModel):
    """Model for status update events"""
    video_id: str = Field(..., description="Twitch video ID")
    stage: ProcessingStage = Field(..., description="Current processing stage")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    estimated_completion_seconds: int = Field(..., description="Estimated seconds until completion")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(..., description="Update timestamp")
    error_details: Optional[str] = Field(None, description="Error details if applicable")


class JobStatus(BaseModel):
    """Model for job status queries"""
    job_id: str = Field(..., description="Unique job identifier")
    video_id: str = Field(..., description="Twitch video ID")
    status: ProcessingStage = Field(..., description="Current processing stage")
    progress: ProgressInfo = Field(..., description="Progress information")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    error: Optional[ErrorInfo] = Field(None, description="Error information if failed")


class WebhookRegistration(BaseModel):
    """Model for webhook registration"""
    video_id: str = Field(..., description="Twitch video ID")
    callback_url: HttpUrl = Field(..., description="Webhook callback URL")
    auth_token: Optional[str] = Field(None, description="Authentication token for webhook")


# Response examples for documentation
ANALYSIS_START_RESPONSE_EXAMPLE = {
    "status": "success",
    "message": "Analysis started successfully",
    "timestamp": "2024-01-15T10:30:00Z",
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "video_id": "2434635255",
    "progress": {
        "percentage": 0.0,
        "current_stage": "Queued",
        "stages_completed": 0,
        "total_stages": 5,
        "estimated_completion_seconds": 3600,
        "estimated_completion_time": "2024-01-15T11:30:00Z",
        "stage_progress": {}
    },
    "metadata": {
        "api_version": "2.0.0",
        "status_url": "/api/v1/analysis/550e8400-e29b-41d4-a716-446655440000/status",
        "stream_url": "/api/v1/analysis/550e8400-e29b-41d4-a716-446655440000/stream"
    }
}

ANALYSIS_ERROR_RESPONSE_EXAMPLE = {
    "status": "failed",
    "message": "Analysis failed due to invalid video URL",
    "timestamp": "2024-01-15T10:30:00Z",
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "video_id": "2434635255",
    "progress": {
        "percentage": 0.0,
        "current_stage": "Failed",
        "stages_completed": 0,
        "total_stages": 5,
        "estimated_completion_seconds": 0,
        "stage_progress": {}
    },
    "error": {
        "error_type": "invalid_video_url",
        "error_code": "INVALID_URL_FORMAT",
        "error_message": "The provided URL is not a valid Twitch VOD URL",
        "error_details": "URL must match pattern: https://www.twitch.tv/videos/{video_id}",
        "is_retryable": False,
        "suggested_action": "Please provide a valid Twitch VOD URL and try again",
        "support_reference": "ERR-2024-0115-001"
    }
}
