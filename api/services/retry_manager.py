"""
Retry Manager Service

This module provides intelligent retry mechanisms with exponential backoff,
circuit breaker patterns, and retry policies for different types of operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

from ..models import ErrorType

# Set up logging
logger = logging.getLogger(__name__)


class RetryPolicy(Enum):
    """Retry policy types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    retryable_errors: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.NETWORK_ERROR,
        ErrorType.EXTERNAL_SERVICE_ERROR,
        ErrorType.PROCESSING_TIMEOUT,
        ErrorType.INSUFFICIENT_RESOURCES
    ])


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    delay_seconds: float
    error: Exception
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
        
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self):
        """Record a successful operation"""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None
        
    def record_failure(self):
        """Record a failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class RetryManager:
    """
    Advanced retry manager with circuit breaker and intelligent backoff
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_history: Dict[str, List[RetryAttempt]] = {}
        logger.info("RetryManager initialized")
    
    def get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an operation"""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker()
        return self.circuit_breakers[operation_name]
    
    async def retry_with_backoff(
        self,
        func: Callable,
        config: Optional[RetryConfig] = None,
        operation_name: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute function with intelligent retry and backoff
        
        Args:
            func: Async function to execute
            config: Retry configuration
            operation_name: Name for circuit breaker tracking
            context: Additional context for logging
            
        Returns:
            Result of the function execution
            
        Raises:
            Exception: The last exception if all retries fail
        """
        if config is None:
            config = RetryConfig()
        
        circuit_breaker = self.get_circuit_breaker(operation_name)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker is open for operation: {operation_name}")
        
        last_exception = None
        retry_attempts = []
        
        for attempt in range(config.max_retries + 1):
            try:
                logger.debug(f"Executing {operation_name}, attempt {attempt + 1}/{config.max_retries + 1}")
                
                result = await func()
                
                # Success - record it and return
                circuit_breaker.record_success()
                
                if retry_attempts:
                    logger.info(f"Operation {operation_name} succeeded after {len(retry_attempts)} retries")
                    self.retry_history[operation_name] = retry_attempts
                
                return result
                
            except Exception as e:
                last_exception = e
                circuit_breaker.record_failure()
                
                # Check if this error type is retryable
                from .error_handler import error_classifier
                error_info = error_classifier.classify_error(e, context)
                
                if not error_info.is_retryable or attempt == config.max_retries:
                    logger.error(f"Operation {operation_name} failed permanently: {str(e)}")
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt, config)
                
                # Record retry attempt
                retry_attempt = RetryAttempt(
                    attempt_number=attempt + 1,
                    delay_seconds=delay,
                    error=e,
                    timestamp=datetime.utcnow(),
                    context=context
                )
                retry_attempts.append(retry_attempt)
                
                logger.warning(
                    f"Operation {operation_name} failed (attempt {attempt + 1}): {str(e)}. "
                    f"Retrying in {delay:.1f}s..."
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries failed
        self.retry_history[operation_name] = retry_attempts
        logger.error(f"Operation {operation_name} failed after {len(retry_attempts)} retries")
        
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"Operation {operation_name} failed with unknown error")
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt based on policy"""
        
        if config.policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.exponential_base ** attempt)
        elif config.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = config.base_delay * (attempt + 1)
        elif config.policy == RetryPolicy.FIXED_DELAY:
            delay = config.base_delay
        else:  # IMMEDIATE
            delay = 0
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter and delay > 0:
            import random
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay
    
    async def retry_with_custom_policy(
        self,
        func: Callable,
        should_retry: Callable[[Exception], bool],
        delays: List[float],
        operation_name: str = "custom",
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute function with custom retry policy
        
        Args:
            func: Async function to execute
            should_retry: Function that determines if an exception should trigger a retry
            delays: List of delays for each retry attempt
            operation_name: Name for tracking
            context: Additional context
            
        Returns:
            Result of the function execution
        """
        circuit_breaker = self.get_circuit_breaker(operation_name)
        
        if not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker is open for operation: {operation_name}")
        
        last_exception = None
        
        for attempt, delay in enumerate([0] + delays):  # First attempt has no delay
            try:
                if delay > 0:
                    logger.debug(f"Waiting {delay}s before retry attempt {attempt}")
                    await asyncio.sleep(delay)
                
                result = await func()
                circuit_breaker.record_success()
                return result
                
            except Exception as e:
                last_exception = e
                circuit_breaker.record_failure()
                
                if attempt == len(delays) or not should_retry(e):
                    break
                
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
        
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"Operation {operation_name} failed with unknown error")
    
    def get_retry_statistics(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get retry statistics for an operation"""
        if operation_name not in self.retry_history:
            return None
        
        attempts = self.retry_history[operation_name]
        if not attempts:
            return None
        
        total_delay = sum(attempt.delay_seconds for attempt in attempts)
        error_types = [type(attempt.error).__name__ for attempt in attempts]
        
        return {
            "operation_name": operation_name,
            "total_attempts": len(attempts),
            "total_delay_seconds": total_delay,
            "error_types": error_types,
            "first_attempt": attempts[0].timestamp.isoformat(),
            "last_attempt": attempts[-1].timestamp.isoformat(),
            "circuit_breaker_state": self.circuit_breakers.get(operation_name, {}).state if operation_name in self.circuit_breakers else "unknown"
        }
    
    def reset_circuit_breaker(self, operation_name: str):
        """Manually reset a circuit breaker"""
        if operation_name in self.circuit_breakers:
            self.circuit_breakers[operation_name].record_success()
            logger.info(f"Circuit breaker reset for operation: {operation_name}")
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all operations"""
        stats = {}
        for operation_name in self.retry_history:
            stats[operation_name] = self.get_retry_statistics(operation_name)
        
        return {
            "operations": stats,
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            }
        }


# Global retry manager instance
retry_manager = RetryManager()


# Convenience functions for common retry patterns
async def retry_network_operation(func: Callable, operation_name: str = "network_op") -> Any:
    """Retry network operations with appropriate config"""
    config = RetryConfig(
        max_retries=3,
        base_delay=2.0,
        max_delay=30.0,
        policy=RetryPolicy.EXPONENTIAL_BACKOFF,
        retryable_errors=[ErrorType.NETWORK_ERROR, ErrorType.EXTERNAL_SERVICE_ERROR]
    )
    return await retry_manager.retry_with_backoff(func, config, operation_name)


async def retry_processing_operation(func: Callable, operation_name: str = "processing_op") -> Any:
    """Retry processing operations with appropriate config"""
    config = RetryConfig(
        max_retries=2,
        base_delay=5.0,
        max_delay=120.0,
        policy=RetryPolicy.EXPONENTIAL_BACKOFF,
        retryable_errors=[ErrorType.PROCESSING_TIMEOUT, ErrorType.INSUFFICIENT_RESOURCES]
    )
    return await retry_manager.retry_with_backoff(func, config, operation_name)


async def retry_external_service(func: Callable, operation_name: str = "external_service") -> Any:
    """Retry external service calls with appropriate config"""
    config = RetryConfig(
        max_retries=4,
        base_delay=1.0,
        max_delay=60.0,
        policy=RetryPolicy.EXPONENTIAL_BACKOFF,
        retryable_errors=[ErrorType.EXTERNAL_SERVICE_ERROR, ErrorType.NETWORK_ERROR]
    )
    return await retry_manager.retry_with_backoff(func, config, operation_name)
