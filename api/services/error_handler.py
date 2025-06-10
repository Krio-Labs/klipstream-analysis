"""
Error Handler Service

This module provides comprehensive error classification, handling, and recovery
suggestions for the KlipStream Analysis API.
"""

import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..models import ErrorType, ErrorInfo

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """Pattern for matching and classifying errors"""
    pattern: str
    error_type: ErrorType
    error_code: str
    is_retryable: bool
    retry_after_seconds: Optional[int] = None
    suggested_action: str = ""


class ErrorClassifier:
    """
    Advanced error classifier that analyzes exceptions and provides
    detailed error information with recovery suggestions
    """
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        logger.info("ErrorClassifier initialized with {} patterns".format(len(self.error_patterns)))
    
    def _initialize_error_patterns(self) -> list[ErrorPattern]:
        """Initialize error patterns for classification"""
        return [
            # Network and connectivity errors
            ErrorPattern(
                pattern=r"(timeout|timed out|connection.*timeout)",
                error_type=ErrorType.NETWORK_ERROR,
                error_code="NETWORK_TIMEOUT",
                is_retryable=True,
                retry_after_seconds=60,
                suggested_action="The request timed out. Please check your internet connection and try again."
            ),
            ErrorPattern(
                pattern=r"(connection.*refused|connection.*failed|network.*unreachable)",
                error_type=ErrorType.NETWORK_ERROR,
                error_code="CONNECTION_FAILED",
                is_retryable=True,
                retry_after_seconds=120,
                suggested_action="Unable to connect to the service. Please try again in a few minutes."
            ),
            ErrorPattern(
                pattern=r"(dns.*resolution.*failed|name.*not.*resolved)",
                error_type=ErrorType.NETWORK_ERROR,
                error_code="DNS_RESOLUTION_FAILED",
                is_retryable=True,
                retry_after_seconds=60,
                suggested_action="DNS resolution failed. Please check your internet connection."
            ),
            
            # Twitch/Video specific errors
            ErrorPattern(
                pattern=r"(invalid.*twitch.*url|not.*valid.*twitch|twitch.*url.*format)",
                error_type=ErrorType.INVALID_VIDEO_URL,
                error_code="INVALID_TWITCH_URL",
                is_retryable=False,
                suggested_action="Please provide a valid Twitch VOD URL in the format: https://www.twitch.tv/videos/{video_id}"
            ),
            ErrorPattern(
                pattern=r"(video.*not.*found|404.*not.*found|video.*does.*not.*exist)",
                error_type=ErrorType.INVALID_VIDEO_URL,
                error_code="VIDEO_NOT_FOUND",
                is_retryable=False,
                suggested_action="The video was not found. It may have been deleted or the URL is incorrect."
            ),
            ErrorPattern(
                pattern=r"(video.*private|video.*unavailable|access.*denied)",
                error_type=ErrorType.INVALID_VIDEO_URL,
                error_code="VIDEO_UNAVAILABLE",
                is_retryable=False,
                suggested_action="The video is private or unavailable. Please ensure the video is publicly accessible."
            ),
            ErrorPattern(
                pattern=r"(video.*too.*large|file.*size.*exceeded|video.*duration.*exceeded)",
                error_type=ErrorType.VIDEO_TOO_LARGE,
                error_code="VIDEO_SIZE_EXCEEDED",
                is_retryable=False,
                suggested_action="The video is too large to process. Please try with a shorter video (under 4 hours)."
            ),
            
            # Processing and resource errors
            ErrorPattern(
                pattern=r"(out.*of.*memory|memory.*error|insufficient.*memory)",
                error_type=ErrorType.INSUFFICIENT_RESOURCES,
                error_code="OUT_OF_MEMORY",
                is_retryable=True,
                retry_after_seconds=300,
                suggested_action="Insufficient memory to process the video. Please try again later when resources are available."
            ),
            ErrorPattern(
                pattern=r"(disk.*space|no.*space.*left|storage.*full)",
                error_type=ErrorType.INSUFFICIENT_RESOURCES,
                error_code="INSUFFICIENT_STORAGE",
                is_retryable=True,
                retry_after_seconds=600,
                suggested_action="Insufficient storage space. Please try again later."
            ),
            ErrorPattern(
                pattern=r"(processing.*timeout|operation.*timed.*out|exceeded.*time.*limit)",
                error_type=ErrorType.PROCESSING_TIMEOUT,
                error_code="PROCESSING_TIMEOUT",
                is_retryable=True,
                retry_after_seconds=300,
                suggested_action="Processing took too long and timed out. Please try again with a shorter video."
            ),
            
            # External service errors
            ErrorPattern(
                pattern=r"(deepgram.*error|transcription.*service.*error|api.*key.*invalid)",
                error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                error_code="TRANSCRIPTION_SERVICE_ERROR",
                is_retryable=True,
                retry_after_seconds=180,
                suggested_action="The transcription service is temporarily unavailable. Please try again in a few minutes."
            ),

            # GPU and transcription specific errors
            ErrorPattern(
                pattern=r"(gpu.*memory.*error|cuda.*out.*of.*memory|gpu.*memory.*exceeded)",
                error_type=ErrorType.INSUFFICIENT_RESOURCES,
                error_code="GPU_MEMORY_ERROR",
                is_retryable=True,
                retry_after_seconds=300,
                suggested_action="GPU memory exceeded. The system will automatically fallback to CPU transcription or try again later."
            ),
            ErrorPattern(
                pattern=r"(model.*loading.*error|model.*not.*found|parakeet.*model.*error)",
                error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                error_code="MODEL_LOADING_ERROR",
                is_retryable=True,
                retry_after_seconds=180,
                suggested_action="Failed to load transcription model. The system will fallback to alternative transcription methods."
            ),
            ErrorPattern(
                pattern=r"(gpu.*not.*available|cuda.*not.*available|no.*gpu.*detected)",
                error_type=ErrorType.INSUFFICIENT_RESOURCES,
                error_code="GPU_NOT_AVAILABLE",
                is_retryable=True,
                retry_after_seconds=60,
                suggested_action="GPU acceleration not available. The system will use CPU-based transcription instead."
            ),
            ErrorPattern(
                pattern=r"(transcription.*fallback.*failed|all.*transcription.*methods.*failed)",
                error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                error_code="TRANSCRIPTION_FALLBACK_FAILED",
                is_retryable=True,
                retry_after_seconds=600,
                suggested_action="All transcription methods failed. Please try again later or contact support."
            ),
            ErrorPattern(
                pattern=r"(parakeet.*timeout|gpu.*transcription.*timeout|model.*inference.*timeout)",
                error_type=ErrorType.PROCESSING_TIMEOUT,
                error_code="TRANSCRIPTION_TIMEOUT",
                is_retryable=True,
                retry_after_seconds=300,
                suggested_action="GPU transcription timed out. The system will fallback to faster transcription methods."
            ),
            ErrorPattern(
                pattern=r"(hybrid.*processing.*error|transcription.*method.*selection.*failed)",
                error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                error_code="HYBRID_TRANSCRIPTION_ERROR",
                is_retryable=True,
                retry_after_seconds=180,
                suggested_action="Hybrid transcription processing failed. The system will try alternative methods."
            ),
            ErrorPattern(
                pattern=r"(convex.*error|database.*error|db.*connection.*failed)",
                error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                error_code="DATABASE_ERROR",
                is_retryable=True,
                retry_after_seconds=120,
                suggested_action="Database service is temporarily unavailable. Please try again shortly."
            ),
            ErrorPattern(
                pattern=r"(gcs.*error|cloud.*storage.*error|bucket.*not.*found)",
                error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                error_code="STORAGE_SERVICE_ERROR",
                is_retryable=True,
                retry_after_seconds=180,
                suggested_action="Cloud storage service is temporarily unavailable. Please try again in a few minutes."
            ),
            ErrorPattern(
                pattern=r"(rate.*limit|too.*many.*requests|quota.*exceeded)",
                error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                error_code="RATE_LIMIT_EXCEEDED",
                is_retryable=True,
                retry_after_seconds=900,  # 15 minutes
                suggested_action="Rate limit exceeded. Please wait 15 minutes before trying again."
            ),
            
            # File and format errors
            ErrorPattern(
                pattern=r"(file.*not.*found|no.*such.*file|path.*does.*not.*exist)",
                error_type=ErrorType.UNKNOWN_ERROR,
                error_code="FILE_NOT_FOUND",
                is_retryable=True,
                retry_after_seconds=60,
                suggested_action="A required file was not found. This may be a temporary issue - please try again."
            ),
            ErrorPattern(
                pattern=r"(permission.*denied|access.*forbidden|unauthorized)",
                error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                error_code="PERMISSION_DENIED",
                is_retryable=False,
                suggested_action="Permission denied. Please contact support if this issue persists."
            ),
            ErrorPattern(
                pattern=r"(invalid.*format|unsupported.*format|format.*not.*supported)",
                error_type=ErrorType.INVALID_VIDEO_URL,
                error_code="UNSUPPORTED_FORMAT",
                is_retryable=False,
                suggested_action="The video format is not supported. Please try with a different video."
            ),
        ]
    
    def classify_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """
        Classify an error and return detailed error information
        
        Args:
            exception: The exception to classify
            context: Optional context information about where the error occurred
            
        Returns:
            ErrorInfo object with classification and recovery suggestions
        """
        error_message = str(exception)
        error_details = f"{type(exception).__name__}: {error_message}"
        
        # Add context information if available
        if context:
            error_details += f"\nContext: {context}"
        
        # Try to match against known patterns
        for pattern in self.error_patterns:
            if re.search(pattern.pattern, error_message, re.IGNORECASE):
                logger.info(f"Classified error as {pattern.error_type.value} using pattern: {pattern.pattern}")
                
                return ErrorInfo(
                    error_type=pattern.error_type,
                    error_code=pattern.error_code,
                    error_message=self._humanize_error_message(pattern.error_type, error_message),
                    error_details=error_details,
                    is_retryable=pattern.is_retryable,
                    retry_after_seconds=pattern.retry_after_seconds,
                    suggested_action=pattern.suggested_action,
                    support_reference=self._generate_support_reference(pattern.error_code)
                )
        
        # Check exception type for additional classification
        error_info = self._classify_by_exception_type(exception, error_details)
        if error_info:
            return error_info
        
        # Default to unknown error
        logger.warning(f"Could not classify error: {error_message}")
        return ErrorInfo(
            error_type=ErrorType.UNKNOWN_ERROR,
            error_code="UNCLASSIFIED_ERROR",
            error_message="An unexpected error occurred during processing",
            error_details=error_details,
            is_retryable=True,
            retry_after_seconds=120,
            suggested_action="An unexpected error occurred. Please try again or contact support if the issue persists.",
            support_reference=self._generate_support_reference("UNCLASSIFIED_ERROR")
        )
    
    def _classify_by_exception_type(self, exception: Exception, error_details: str) -> Optional[ErrorInfo]:
        """Classify error based on exception type"""
        
        # Import here to avoid circular imports
        import aiohttp
        import asyncio
        
        if isinstance(exception, aiohttp.ClientTimeout):
            return ErrorInfo(
                error_type=ErrorType.NETWORK_ERROR,
                error_code="HTTP_TIMEOUT",
                error_message="Request timed out while communicating with external service",
                error_details=error_details,
                is_retryable=True,
                retry_after_seconds=60,
                suggested_action="The request timed out. Please try again.",
                support_reference=self._generate_support_reference("HTTP_TIMEOUT")
            )
        
        elif isinstance(exception, aiohttp.ClientResponseError):
            if exception.status == 404:
                return ErrorInfo(
                    error_type=ErrorType.INVALID_VIDEO_URL,
                    error_code="HTTP_404_NOT_FOUND",
                    error_message="The requested resource was not found",
                    error_details=error_details,
                    is_retryable=False,
                    suggested_action="The video URL is invalid or the video has been deleted. Please check the URL and try again.",
                    support_reference=self._generate_support_reference("HTTP_404_NOT_FOUND")
                )
            elif exception.status == 403:
                return ErrorInfo(
                    error_type=ErrorType.INVALID_VIDEO_URL,
                    error_code="HTTP_403_FORBIDDEN",
                    error_message="Access to the resource is forbidden",
                    error_details=error_details,
                    is_retryable=False,
                    suggested_action="The video is private or access is restricted. Please ensure the video is publicly accessible.",
                    support_reference=self._generate_support_reference("HTTP_403_FORBIDDEN")
                )
            elif exception.status >= 500:
                return ErrorInfo(
                    error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                    error_code=f"HTTP_{exception.status}_SERVER_ERROR",
                    error_message="External service is temporarily unavailable",
                    error_details=error_details,
                    is_retryable=True,
                    retry_after_seconds=300,
                    suggested_action="The external service is temporarily unavailable. Please try again in a few minutes.",
                    support_reference=self._generate_support_reference(f"HTTP_{exception.status}_SERVER_ERROR")
                )
        
        elif isinstance(exception, asyncio.TimeoutError):
            return ErrorInfo(
                error_type=ErrorType.PROCESSING_TIMEOUT,
                error_code="ASYNC_TIMEOUT",
                error_message="Operation timed out",
                error_details=error_details,
                is_retryable=True,
                retry_after_seconds=180,
                suggested_action="The operation timed out. Please try again with a shorter video or try again later.",
                support_reference=self._generate_support_reference("ASYNC_TIMEOUT")
            )
        
        elif isinstance(exception, MemoryError):
            return ErrorInfo(
                error_type=ErrorType.INSUFFICIENT_RESOURCES,
                error_code="MEMORY_ERROR",
                error_message="Insufficient memory to complete the operation",
                error_details=error_details,
                is_retryable=True,
                retry_after_seconds=600,
                suggested_action="Insufficient memory to process the video. Please try again later or with a shorter video.",
                support_reference=self._generate_support_reference("MEMORY_ERROR")
            )
        
        elif isinstance(exception, FileNotFoundError):
            return ErrorInfo(
                error_type=ErrorType.UNKNOWN_ERROR,
                error_code="FILE_NOT_FOUND_ERROR",
                error_message="Required file not found",
                error_details=error_details,
                is_retryable=True,
                retry_after_seconds=60,
                suggested_action="A required file was not found. This may be a temporary issue - please try again.",
                support_reference=self._generate_support_reference("FILE_NOT_FOUND_ERROR")
            )
        
        return None

    def classify_transcription_error(self, exception: Exception, transcription_context: Dict[str, Any]) -> ErrorInfo:
        """
        Specialized error classification for transcription-related errors

        Args:
            exception: The exception to classify
            transcription_context: Context about transcription method, GPU usage, etc.

        Returns:
            ErrorInfo object with transcription-specific classification
        """
        error_message = str(exception)
        method_used = transcription_context.get("method_used", "unknown")
        gpu_available = transcription_context.get("gpu_available", False)
        fallback_attempted = transcription_context.get("fallback_attempted", False)

        # Enhanced context for transcription errors
        enhanced_context = {
            **transcription_context,
            "transcription_method": method_used,
            "gpu_available": gpu_available,
            "fallback_attempted": fallback_attempted
        }

        # Check for specific transcription error patterns first
        transcription_patterns = [
            (r"(gpu.*memory|cuda.*memory)", "GPU_MEMORY_ERROR",
             "GPU memory insufficient for transcription. Falling back to CPU or smaller batch size."),
            (r"(model.*loading|model.*download)", "MODEL_LOADING_ERROR",
             "Failed to load transcription model. Trying alternative transcription method."),
            (r"(parakeet.*error|nemo.*error)", "PARAKEET_ERROR",
             "GPU transcription model error. Falling back to cloud-based transcription."),
            (r"(deepgram.*quota|deepgram.*limit)", "DEEPGRAM_QUOTA_ERROR",
             "Deepgram API quota exceeded. Please try again later or contact support."),
            (r"(hybrid.*split|hybrid.*merge)", "HYBRID_PROCESSING_ERROR",
             "Hybrid transcription processing failed. Trying single-method transcription."),
        ]

        for pattern, error_code, suggested_action in transcription_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return ErrorInfo(
                    error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
                    error_code=error_code,
                    error_message=f"Transcription error with {method_used} method",
                    error_details=f"{type(exception).__name__}: {error_message}\nContext: {enhanced_context}",
                    is_retryable=True,
                    retry_after_seconds=180,
                    suggested_action=suggested_action,
                    support_reference=self._generate_support_reference(error_code)
                )

        # Fall back to general classification with enhanced context
        return self.classify_error(exception, enhanced_context)

    def _humanize_error_message(self, error_type: ErrorType, original_message: str) -> str:
        """Convert technical error messages to user-friendly messages"""
        
        humanized_messages = {
            ErrorType.NETWORK_ERROR: "Network connection issue occurred",
            ErrorType.INVALID_VIDEO_URL: "Invalid or inaccessible video URL",
            ErrorType.VIDEO_TOO_LARGE: "Video is too large to process",
            ErrorType.PROCESSING_TIMEOUT: "Processing took too long and timed out",
            ErrorType.INSUFFICIENT_RESOURCES: "Insufficient system resources",
            ErrorType.EXTERNAL_SERVICE_ERROR: "External service temporarily unavailable",
            ErrorType.UNKNOWN_ERROR: "An unexpected error occurred"
        }
        
        return humanized_messages.get(error_type, original_message)
    
    def _generate_support_reference(self, error_code: str) -> str:
        """Generate a unique support reference for the error"""
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return f"ERR-{timestamp}-{error_code[:8]}"


# Global error classifier instance
error_classifier = ErrorClassifier()
