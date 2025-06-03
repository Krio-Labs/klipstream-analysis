"""
Webhook API Routes

This module provides REST API endpoints for webhook management,
including registration, configuration, testing, and monitoring.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, HttpUrl, validator

from ..services.webhook_manager import (
    webhook_manager, 
    WebhookEvent, 
    WebhookConfig, 
    WebhookDelivery,
    WebhookStatus
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


# Pydantic models for request/response
class WebhookEventModel(BaseModel):
    """Webhook event model"""
    value: str
    
    @validator('value')
    def validate_event(cls, v):
        try:
            WebhookEvent(v)
            return v
        except ValueError:
            valid_events = [e.value for e in WebhookEvent]
            raise ValueError(f"Invalid event. Must be one of: {valid_events}")


class WebhookCreateRequest(BaseModel):
    """Request model for creating a webhook"""
    url: HttpUrl
    events: List[str]
    secret: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    
    @validator('events')
    def validate_events(cls, v):
        valid_events = [e.value for e in WebhookEvent]
        for event in v:
            if event not in valid_events:
                raise ValueError(f"Invalid event '{event}'. Must be one of: {valid_events}")
        return v
    
    @validator('timeout_seconds')
    def validate_timeout(cls, v):
        if v < 1 or v > 300:
            raise ValueError("Timeout must be between 1 and 300 seconds")
        return v
    
    @validator('max_retries')
    def validate_retries(cls, v):
        if v < 0 or v > 10:
            raise ValueError("Max retries must be between 0 and 10")
        return v


class WebhookUpdateRequest(BaseModel):
    """Request model for updating a webhook"""
    url: Optional[HttpUrl] = None
    events: Optional[List[str]] = None
    secret: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    is_active: Optional[bool] = None
    
    @validator('events')
    def validate_events(cls, v):
        if v is not None:
            valid_events = [e.value for e in WebhookEvent]
            for event in v:
                if event not in valid_events:
                    raise ValueError(f"Invalid event '{event}'. Must be one of: {valid_events}")
        return v


class WebhookResponse(BaseModel):
    """Response model for webhook information"""
    id: str
    url: str
    events: List[str]
    timeout_seconds: int
    max_retries: int
    is_active: bool
    created_at: str
    updated_at: str
    
    @classmethod
    def from_webhook_config(cls, webhook: WebhookConfig) -> "WebhookResponse":
        return cls(
            id=webhook.id,
            url=webhook.url,
            events=[e.value for e in webhook.events],
            timeout_seconds=webhook.timeout_seconds,
            max_retries=webhook.max_retries,
            is_active=webhook.is_active,
            created_at=webhook.created_at.isoformat(),
            updated_at=webhook.updated_at.isoformat()
        )


class WebhookDeliveryResponse(BaseModel):
    """Response model for webhook delivery information"""
    id: str
    webhook_id: str
    event: str
    status: str
    attempt_count: int
    response_status: Optional[int]
    error_message: Optional[str]
    created_at: str
    last_attempt_at: Optional[str]
    next_retry_at: Optional[str]
    
    @classmethod
    def from_delivery(cls, delivery: WebhookDelivery) -> "WebhookDeliveryResponse":
        return cls(
            id=delivery.id,
            webhook_id=delivery.webhook_id,
            event=delivery.event.value,
            status=delivery.status.value,
            attempt_count=delivery.attempt_count,
            response_status=delivery.response_status,
            error_message=delivery.error_message,
            created_at=delivery.created_at.isoformat(),
            last_attempt_at=delivery.last_attempt_at.isoformat() if delivery.last_attempt_at else None,
            next_retry_at=delivery.next_retry_at.isoformat() if delivery.next_retry_at else None
        )


class WebhookStatsResponse(BaseModel):
    """Response model for webhook statistics"""
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    pending_deliveries: int
    success_rate: float


class WebhookTestRequest(BaseModel):
    """Request model for testing a webhook"""
    event: str = "job.started"
    test_payload: Optional[Dict[str, Any]] = None
    
    @validator('event')
    def validate_event(cls, v):
        try:
            WebhookEvent(v)
            return v
        except ValueError:
            valid_events = [e.value for e in WebhookEvent]
            raise ValueError(f"Invalid event. Must be one of: {valid_events}")


# API Endpoints

@router.post("/", response_model=WebhookResponse, status_code=201)
async def create_webhook(
    webhook_id: str = Query(..., description="Unique webhook identifier"),
    request: WebhookCreateRequest = Body(...)
):
    """
    Register a new webhook
    
    Creates a new webhook configuration for receiving event notifications.
    The webhook will receive HTTP POST requests for subscribed events.
    """
    try:
        # Check if webhook ID already exists
        if webhook_manager.get_webhook(webhook_id):
            raise HTTPException(
                status_code=409,
                detail=f"Webhook with ID '{webhook_id}' already exists"
            )
        
        # Convert string events to WebhookEvent enums
        events = [WebhookEvent(event) for event in request.events]
        
        # Register webhook
        webhook = webhook_manager.register_webhook(
            webhook_id=webhook_id,
            url=str(request.url),
            events=events,
            secret=request.secret,
            headers=request.headers,
            timeout_seconds=request.timeout_seconds,
            max_retries=request.max_retries
        )
        
        logger.info(f"Created webhook {webhook_id} for {len(events)} events")
        
        return WebhookResponse.from_webhook_config(webhook)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating webhook {webhook_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=List[WebhookResponse])
async def list_webhooks(
    active_only: bool = Query(True, description="Only return active webhooks")
):
    """
    List all registered webhooks
    
    Returns a list of all webhook configurations, optionally filtered to active webhooks only.
    """
    try:
        webhooks = webhook_manager.list_webhooks(active_only=active_only)
        return [WebhookResponse.from_webhook_config(webhook) for webhook in webhooks]
        
    except Exception as e:
        logger.error(f"Error listing webhooks: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(webhook_id: str):
    """
    Get webhook configuration
    
    Returns the configuration for a specific webhook.
    """
    try:
        webhook = webhook_manager.get_webhook(webhook_id)
        if not webhook:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook '{webhook_id}' not found"
            )
        
        return WebhookResponse.from_webhook_config(webhook)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting webhook {webhook_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: str,
    request: WebhookUpdateRequest
):
    """
    Update webhook configuration
    
    Updates the configuration for an existing webhook.
    """
    try:
        # Check if webhook exists
        if not webhook_manager.get_webhook(webhook_id):
            raise HTTPException(
                status_code=404,
                detail=f"Webhook '{webhook_id}' not found"
            )
        
        # Convert string events to WebhookEvent enums if provided
        events = None
        if request.events is not None:
            events = [WebhookEvent(event) for event in request.events]
        
        # Update webhook
        webhook = webhook_manager.update_webhook(
            webhook_id=webhook_id,
            url=str(request.url) if request.url else None,
            events=events,
            secret=request.secret,
            headers=request.headers,
            is_active=request.is_active
        )
        
        if not webhook:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook '{webhook_id}' not found"
            )
        
        logger.info(f"Updated webhook {webhook_id}")
        
        return WebhookResponse.from_webhook_config(webhook)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating webhook {webhook_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{webhook_id}", status_code=204)
async def delete_webhook(webhook_id: str):
    """
    Delete webhook
    
    Removes a webhook configuration. This will stop all future notifications to this webhook.
    """
    try:
        success = webhook_manager.delete_webhook(webhook_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook '{webhook_id}' not found"
            )
        
        logger.info(f"Deleted webhook {webhook_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting webhook {webhook_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{webhook_id}/logs", response_model=List[WebhookDeliveryResponse])
async def get_webhook_logs(
    webhook_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return")
):
    """
    Get webhook delivery logs
    
    Returns the delivery history for a specific webhook, including success/failure status.
    """
    try:
        # Check if webhook exists
        if not webhook_manager.get_webhook(webhook_id):
            raise HTTPException(
                status_code=404,
                detail=f"Webhook '{webhook_id}' not found"
            )
        
        logs = webhook_manager.get_delivery_logs(webhook_id, limit=limit)
        return [WebhookDeliveryResponse.from_delivery(log) for log in logs]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting webhook logs for {webhook_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{webhook_id}/stats", response_model=WebhookStatsResponse)
async def get_webhook_stats(webhook_id: str):
    """
    Get webhook delivery statistics
    
    Returns delivery statistics for a specific webhook, including success rates.
    """
    try:
        # Check if webhook exists
        if not webhook_manager.get_webhook(webhook_id):
            raise HTTPException(
                status_code=404,
                detail=f"Webhook '{webhook_id}' not found"
            )
        
        stats = webhook_manager.get_delivery_stats(webhook_id)
        return WebhookStatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting webhook stats for {webhook_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{webhook_id}/test", status_code=202)
async def test_webhook(
    webhook_id: str,
    request: WebhookTestRequest = Body(...)
):
    """
    Test webhook delivery
    
    Sends a test event to the webhook to verify configuration and connectivity.
    """
    try:
        # Check if webhook exists
        webhook = webhook_manager.get_webhook(webhook_id)
        if not webhook:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook '{webhook_id}' not found"
            )
        
        # Create test payload
        test_payload = request.test_payload or {
            "message": "This is a test webhook delivery",
            "test": True,
            "timestamp": "2024-01-15T12:00:00Z"
        }
        
        # Send test webhook
        await webhook_manager.send_webhook(
            event=WebhookEvent(request.event),
            job_id="test-job-id",
            video_id="test-video-id",
            payload=test_payload,
            webhook_ids=[webhook_id]
        )
        
        logger.info(f"Sent test webhook to {webhook_id}")
        
        return {
            "message": f"Test webhook sent to {webhook_id}",
            "event": request.event,
            "webhook_url": webhook.url
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error testing webhook {webhook_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/events/available", response_model=List[Dict[str, str]])
async def get_available_events():
    """
    Get available webhook events
    
    Returns a list of all available webhook events that can be subscribed to.
    """
    try:
        events = [
            {
                "value": event.value,
                "description": _get_event_description(event)
            }
            for event in WebhookEvent
        ]
        
        return events
        
    except Exception as e:
        logger.error(f"Error getting available events: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _get_event_description(event: WebhookEvent) -> str:
    """Get human-readable description for webhook events"""
    descriptions = {
        WebhookEvent.JOB_STARTED: "Triggered when a video analysis job is started",
        WebhookEvent.JOB_PROGRESS: "Triggered when job progress is updated (every 10% progress)",
        WebhookEvent.JOB_COMPLETED: "Triggered when a job completes successfully",
        WebhookEvent.JOB_FAILED: "Triggered when a job fails",
        WebhookEvent.JOB_CANCELLED: "Triggered when a job is cancelled",
        WebhookEvent.JOB_STAGE_CHANGED: "Triggered when job moves to a new processing stage"
    }
    return descriptions.get(event, "No description available")
