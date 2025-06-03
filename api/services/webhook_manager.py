"""
Webhook Manager Service

This module provides comprehensive webhook support for the KlipStream Analysis API,
including registration, delivery, retry logic, and security features.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse

import aiohttp
from ..models import ProcessingStage

# Set up logging
logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Webhook event types"""
    JOB_STARTED = "job.started"
    JOB_PROGRESS = "job.progress"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"
    JOB_STAGE_CHANGED = "job.stage_changed"


class WebhookStatus(Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    id: str
    url: str
    events: Set[WebhookEvent]
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 60
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt"""
    id: str
    webhook_id: str
    event: WebhookEvent
    payload: Dict
    status: WebhookStatus
    attempt_count: int = 0
    last_attempt_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class WebhookManager:
    """
    Comprehensive webhook management system with delivery, retry, and security
    """
    
    def __init__(self, secret_key: str = "klipstream-webhook-secret"):
        self.secret_key = secret_key
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.is_processing = False
        logger.info("WebhookManager initialized")
    
    def register_webhook(
        self,
        webhook_id: str,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 30,
        max_retries: int = 3
    ) -> WebhookConfig:
        """
        Register a new webhook
        
        Args:
            webhook_id: Unique webhook identifier
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional secret for signature verification
            headers: Optional custom headers
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
            
        Returns:
            WebhookConfig object
        """
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid webhook URL: {url}")
        
        if parsed_url.scheme not in ['http', 'https']:
            raise ValueError(f"Webhook URL must use HTTP or HTTPS: {url}")
        
        # Create webhook config
        webhook = WebhookConfig(
            id=webhook_id,
            url=url,
            events=set(events),
            secret=secret,
            headers=headers or {},
            timeout_seconds=timeout_seconds,
            max_retries=max_retries
        )
        
        self.webhooks[webhook_id] = webhook
        logger.info(f"Registered webhook {webhook_id} for {len(events)} events: {url}")
        
        return webhook
    
    def update_webhook(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[WebhookEvent]] = None,
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        is_active: Optional[bool] = None
    ) -> Optional[WebhookConfig]:
        """Update an existing webhook"""
        if webhook_id not in self.webhooks:
            return None
        
        webhook = self.webhooks[webhook_id]
        
        if url is not None:
            # Validate new URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid webhook URL: {url}")
            webhook.url = url
        
        if events is not None:
            webhook.events = set(events)
        
        if secret is not None:
            webhook.secret = secret
        
        if headers is not None:
            webhook.headers = headers
        
        if is_active is not None:
            webhook.is_active = is_active
        
        webhook.updated_at = datetime.utcnow()
        
        logger.info(f"Updated webhook {webhook_id}")
        return webhook
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Deleted webhook {webhook_id}")
            return True
        return False
    
    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook configuration"""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self, active_only: bool = True) -> List[WebhookConfig]:
        """List all webhooks"""
        webhooks = list(self.webhooks.values())
        if active_only:
            webhooks = [w for w in webhooks if w.is_active]
        return webhooks
    
    async def send_webhook(
        self,
        event: WebhookEvent,
        job_id: str,
        video_id: str,
        payload: Dict,
        webhook_ids: Optional[List[str]] = None
    ):
        """
        Send webhook notifications for an event
        
        Args:
            event: Webhook event type
            job_id: Job identifier
            video_id: Video identifier
            payload: Event payload
            webhook_ids: Optional list of specific webhook IDs to notify
        """
        # Find webhooks that should receive this event
        target_webhooks = []
        
        if webhook_ids:
            # Send to specific webhooks
            for webhook_id in webhook_ids:
                if webhook_id in self.webhooks:
                    webhook = self.webhooks[webhook_id]
                    if webhook.is_active and event in webhook.events:
                        target_webhooks.append(webhook)
        else:
            # Send to all subscribed webhooks
            for webhook in self.webhooks.values():
                if webhook.is_active and event in webhook.events:
                    target_webhooks.append(webhook)
        
        if not target_webhooks:
            logger.debug(f"No webhooks registered for event {event.value}")
            return
        
        # Create delivery records and queue them
        for webhook in target_webhooks:
            delivery_id = f"{webhook.id}_{job_id}_{int(time.time())}"
            
            # Prepare payload with metadata
            full_payload = {
                "event": event.value,
                "job_id": job_id,
                "video_id": video_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": payload
            }
            
            delivery = WebhookDelivery(
                id=delivery_id,
                webhook_id=webhook.id,
                event=event,
                payload=full_payload,
                status=WebhookStatus.PENDING
            )
            
            self.deliveries[delivery_id] = delivery
            await self.delivery_queue.put(delivery_id)
            
            logger.debug(f"Queued webhook delivery {delivery_id} for {webhook.url}")
        
        # Start processing if not already running
        if not self.is_processing:
            asyncio.create_task(self._process_delivery_queue())
    
    async def _process_delivery_queue(self):
        """Process webhook delivery queue"""
        if self.is_processing:
            return
        
        self.is_processing = True
        logger.info("Started webhook delivery processing")
        
        try:
            while True:
                try:
                    # Wait for delivery with timeout
                    delivery_id = await asyncio.wait_for(
                        self.delivery_queue.get(),
                        timeout=5.0
                    )
                    
                    if delivery_id in self.deliveries:
                        await self._deliver_webhook(delivery_id)
                    
                except asyncio.TimeoutError:
                    # Check for retries
                    await self._process_retries()
                    
                    # If queue is empty and no retries pending, stop processing
                    if self.delivery_queue.empty() and not self._has_pending_retries():
                        break
                
        except Exception as e:
            logger.error(f"Error in webhook delivery processing: {str(e)}")
        finally:
            self.is_processing = False
            logger.info("Stopped webhook delivery processing")
    
    async def _deliver_webhook(self, delivery_id: str):
        """Deliver a single webhook"""
        delivery = self.deliveries.get(delivery_id)
        if not delivery:
            return
        
        webhook = self.webhooks.get(delivery.webhook_id)
        if not webhook or not webhook.is_active:
            delivery.status = WebhookStatus.CANCELLED
            delivery.error_message = "Webhook not found or inactive"
            return
        
        delivery.attempt_count += 1
        delivery.last_attempt_at = datetime.utcnow()
        delivery.status = WebhookStatus.RETRYING if delivery.attempt_count > 1 else WebhookStatus.PENDING
        
        try:
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "KlipStream-Webhook/2.0",
                **webhook.headers
            }
            
            # Add signature if secret is provided
            payload_json = json.dumps(delivery.payload, separators=(',', ':'))
            if webhook.secret:
                signature = self._generate_signature(payload_json, webhook.secret)
                headers["X-KlipStream-Signature"] = signature
            
            # Make HTTP request
            timeout = aiohttp.ClientTimeout(total=webhook.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    webhook.url,
                    data=payload_json,
                    headers=headers
                ) as response:
                    delivery.response_status = response.status
                    delivery.response_body = await response.text()
                    
                    if 200 <= response.status < 300:
                        delivery.status = WebhookStatus.DELIVERED
                        logger.info(f"Webhook delivered successfully: {delivery_id}")
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}"
                        )
        
        except Exception as e:
            delivery.error_message = str(e)
            logger.warning(f"Webhook delivery failed: {delivery_id} - {str(e)}")
            
            # Schedule retry if attempts remaining
            if delivery.attempt_count < webhook.max_retries:
                retry_delay = webhook.retry_delay_seconds * (2 ** (delivery.attempt_count - 1))
                delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=retry_delay)
                delivery.status = WebhookStatus.RETRYING
                logger.info(f"Scheduled retry for {delivery_id} in {retry_delay} seconds")
            else:
                delivery.status = WebhookStatus.FAILED
                logger.error(f"Webhook delivery failed permanently: {delivery_id}")
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        expected_signature = self._generate_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)
    
    async def _process_retries(self):
        """Process pending retries"""
        now = datetime.utcnow()
        
        for delivery in self.deliveries.values():
            if (delivery.status == WebhookStatus.RETRYING and 
                delivery.next_retry_at and 
                delivery.next_retry_at <= now):
                
                await self.delivery_queue.put(delivery.id)
    
    def _has_pending_retries(self) -> bool:
        """Check if there are pending retries"""
        return any(
            delivery.status == WebhookStatus.RETRYING
            for delivery in self.deliveries.values()
        )
    
    def get_delivery_logs(self, webhook_id: str, limit: int = 100) -> List[WebhookDelivery]:
        """Get delivery logs for a webhook"""
        logs = [
            delivery for delivery in self.deliveries.values()
            if delivery.webhook_id == webhook_id
        ]
        
        # Sort by creation time, most recent first
        logs.sort(key=lambda x: x.created_at, reverse=True)
        
        return logs[:limit]
    
    def get_delivery_stats(self, webhook_id: str) -> Dict:
        """Get delivery statistics for a webhook"""
        deliveries = [
            delivery for delivery in self.deliveries.values()
            if delivery.webhook_id == webhook_id
        ]
        
        if not deliveries:
            return {
                "total_deliveries": 0,
                "successful_deliveries": 0,
                "failed_deliveries": 0,
                "pending_deliveries": 0,
                "success_rate": 0.0
            }
        
        total = len(deliveries)
        successful = len([d for d in deliveries if d.status == WebhookStatus.DELIVERED])
        failed = len([d for d in deliveries if d.status == WebhookStatus.FAILED])
        pending = len([d for d in deliveries if d.status in [WebhookStatus.PENDING, WebhookStatus.RETRYING]])
        
        return {
            "total_deliveries": total,
            "successful_deliveries": successful,
            "failed_deliveries": failed,
            "pending_deliveries": pending,
            "success_rate": (successful / total) * 100 if total > 0 else 0.0
        }


# Global webhook manager instance
webhook_manager = WebhookManager()
