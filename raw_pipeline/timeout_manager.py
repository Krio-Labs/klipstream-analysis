"""
Timeout Management System

This module provides intelligent timeout management for download processes
with adaptive timeouts based on file size, network conditions, and progress.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TimeoutReason(Enum):
    """Reasons for timeout"""
    TOTAL_TIMEOUT = "total_timeout"
    PROGRESS_STALL = "progress_stall"
    NETWORK_TIMEOUT = "network_timeout"
    RESOURCE_TIMEOUT = "resource_timeout"

@dataclass
class TimeoutConfig:
    """Configuration for timeout management"""
    base_timeout_seconds: int = 1800  # 30 minutes base
    max_timeout_seconds: int = 3600   # 1 hour maximum
    progress_stall_timeout: int = 300  # 5 minutes without progress
    network_timeout: int = 60         # 1 minute for network operations
    adaptive_scaling: bool = True     # Enable adaptive timeout scaling
    
@dataclass
class ProgressInfo:
    """Progress information for timeout calculations"""
    percentage: float
    bytes_downloaded: int
    total_bytes: Optional[int]
    last_update: datetime
    download_speed_mbps: float = 0.0
    
@dataclass
class TimeoutEvent:
    """Timeout event information"""
    reason: TimeoutReason
    timeout_seconds: int
    elapsed_seconds: float
    progress_info: Optional[ProgressInfo]
    message: str
    is_recoverable: bool

class AdaptiveTimeoutManager:
    """Manages adaptive timeouts based on progress and conditions"""
    
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.start_time = datetime.now(timezone.utc)
        self.last_progress_time = self.start_time
        self.progress_history = []
        self.current_timeout = config.base_timeout_seconds
        self.timeout_callbacks = []
        
    def register_timeout_callback(self, callback: Callable[[TimeoutEvent], None]):
        """Register a callback to be called when timeout occurs"""
        self.timeout_callbacks.append(callback)
        
    def update_progress(self, progress: ProgressInfo):
        """Update progress information and recalculate timeouts"""
        current_time = datetime.now(timezone.utc)
        
        # Check if progress has actually changed
        if self.progress_history:
            last_progress = self.progress_history[-1]
            if progress.percentage > last_progress.percentage:
                self.last_progress_time = current_time
        else:
            self.last_progress_time = current_time
            
        # Add to history
        self.progress_history.append(progress)
        
        # Keep only recent history (last 10 updates)
        if len(self.progress_history) > 10:
            self.progress_history = self.progress_history[-10:]
            
        # Recalculate adaptive timeout
        if self.config.adaptive_scaling:
            self._calculate_adaptive_timeout(progress)
            
    def _calculate_adaptive_timeout(self, progress: ProgressInfo):
        """Calculate adaptive timeout based on current conditions"""
        
        # Base calculation on estimated completion time
        if progress.total_bytes and progress.download_speed_mbps > 0:
            remaining_bytes = progress.total_bytes - progress.bytes_downloaded
            remaining_mb = remaining_bytes / (1024 * 1024)
            estimated_time = remaining_mb / progress.download_speed_mbps
            
            # Add 50% buffer for safety
            adaptive_timeout = int(estimated_time * 1.5)
            
            # Ensure within bounds
            adaptive_timeout = max(self.config.base_timeout_seconds, adaptive_timeout)
            adaptive_timeout = min(self.config.max_timeout_seconds, adaptive_timeout)
            
            self.current_timeout = adaptive_timeout
            logger.debug(f"Adaptive timeout calculated: {adaptive_timeout}s based on {progress.download_speed_mbps:.2f} Mbps")
        
        # Adjust based on progress rate
        if len(self.progress_history) >= 2:
            recent_progress = self.progress_history[-2:]
            time_diff = (recent_progress[1].last_update - recent_progress[0].last_update).total_seconds()
            progress_diff = recent_progress[1].percentage - recent_progress[0].percentage
            
            if time_diff > 0:
                progress_rate = progress_diff / time_diff  # percentage per second
                
                # If progress is very slow, increase timeout
                if progress_rate < 0.01:  # Less than 1% per 100 seconds
                    self.current_timeout = min(self.config.max_timeout_seconds, self.current_timeout * 1.2)
                    logger.debug(f"Slow progress detected, increased timeout to {self.current_timeout}s")
    
    def check_timeouts(self, current_progress: Optional[ProgressInfo] = None) -> Optional[TimeoutEvent]:
        """Check for various timeout conditions"""
        current_time = datetime.now(timezone.utc)
        elapsed_total = (current_time - self.start_time).total_seconds()
        elapsed_since_progress = (current_time - self.last_progress_time).total_seconds()
        
        # Check total timeout
        if elapsed_total > self.current_timeout:
            event = TimeoutEvent(
                reason=TimeoutReason.TOTAL_TIMEOUT,
                timeout_seconds=self.current_timeout,
                elapsed_seconds=elapsed_total,
                progress_info=current_progress,
                message=f"Total timeout exceeded: {elapsed_total:.0f}s > {self.current_timeout}s",
                is_recoverable=False
            )
            self._notify_timeout(event)
            return event
            
        # Check progress stall timeout
        if elapsed_since_progress > self.config.progress_stall_timeout:
            event = TimeoutEvent(
                reason=TimeoutReason.PROGRESS_STALL,
                timeout_seconds=self.config.progress_stall_timeout,
                elapsed_seconds=elapsed_since_progress,
                progress_info=current_progress,
                message=f"Progress stalled: {elapsed_since_progress:.0f}s without progress",
                is_recoverable=True
            )
            self._notify_timeout(event)
            return event
            
        return None
    
    def _notify_timeout(self, event: TimeoutEvent):
        """Notify all registered callbacks about timeout"""
        for callback in self.timeout_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in timeout callback: {e}")
    
    def get_remaining_time(self) -> int:
        """Get remaining time before timeout"""
        current_time = datetime.now(timezone.utc)
        elapsed = (current_time - self.start_time).total_seconds()
        return max(0, int(self.current_timeout - elapsed))
    
    def get_timeout_status(self) -> Dict[str, Any]:
        """Get current timeout status"""
        current_time = datetime.now(timezone.utc)
        elapsed_total = (current_time - self.start_time).total_seconds()
        elapsed_since_progress = (current_time - self.last_progress_time).total_seconds()
        
        return {
            "current_timeout_seconds": self.current_timeout,
            "elapsed_total_seconds": elapsed_total,
            "elapsed_since_progress_seconds": elapsed_since_progress,
            "remaining_seconds": self.get_remaining_time(),
            "progress_stall_remaining": max(0, self.config.progress_stall_timeout - elapsed_since_progress),
            "timeout_risk_level": self._calculate_risk_level(elapsed_total, elapsed_since_progress)
        }
    
    def _calculate_risk_level(self, elapsed_total: float, elapsed_since_progress: float) -> str:
        """Calculate timeout risk level"""
        total_risk = elapsed_total / self.current_timeout
        progress_risk = elapsed_since_progress / self.config.progress_stall_timeout
        
        max_risk = max(total_risk, progress_risk)
        
        if max_risk < 0.5:
            return "low"
        elif max_risk < 0.8:
            return "medium"
        elif max_risk < 0.95:
            return "high"
        else:
            return "critical"

class TimeoutAwareProcess:
    """Wrapper for processes with timeout management"""
    
    def __init__(self, timeout_manager: AdaptiveTimeoutManager):
        self.timeout_manager = timeout_manager
        self.process = None
        self.monitoring_task = None
        
    async def start_process(self, command, **kwargs):
        """Start process with timeout monitoring"""
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs
        )
        
        # Start timeout monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_timeouts())
        
        return self.process
    
    async def _monitor_timeouts(self):
        """Monitor for timeout conditions"""
        while self.process and self.process.returncode is None:
            timeout_event = self.timeout_manager.check_timeouts()
            
            if timeout_event:
                logger.warning(f"Timeout detected: {timeout_event.message}")
                
                if timeout_event.reason == TimeoutReason.TOTAL_TIMEOUT:
                    # Hard timeout - terminate process
                    await self._terminate_process()
                    break
                elif timeout_event.reason == TimeoutReason.PROGRESS_STALL:
                    # Soft timeout - log warning but continue
                    logger.warning("Progress stalled but continuing...")
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def _terminate_process(self):
        """Gracefully terminate the process"""
        if self.process and self.process.returncode is None:
            logger.info("Terminating process due to timeout...")
            
            # Try graceful termination first
            self.process.terminate()
            
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
                logger.info("Process terminated gracefully")
            except asyncio.TimeoutError:
                # Force kill if graceful termination fails
                logger.warning("Graceful termination failed, force killing process")
                self.process.kill()
                await self.process.wait()
                logger.info("Process force killed")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.process and self.process.returncode is None:
            await self._terminate_process()
