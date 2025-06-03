"""
Metrics Manager Service

This module provides comprehensive metrics collection and monitoring
for the KlipStream Analysis API, including performance, business, and system metrics.
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type categories"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Individual metric value with timestamp"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition and storage"""
    name: str
    metric_type: MetricType
    description: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_value(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a value to the metric"""
        metric_value = MetricValue(
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {}
        )
        self.values.append(metric_value)
    
    def get_current_value(self) -> Optional[float]:
        """Get the most recent value"""
        if self.values:
            return self.values[-1].value
        return None
    
    def get_average(self, minutes: int = 5) -> Optional[float]:
        """Get average value over the last N minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_values = [
            v.value for v in self.values
            if v.timestamp >= cutoff_time
        ]
        
        if recent_values:
            return sum(recent_values) / len(recent_values)
        return None


class MetricsManager:
    """
    Comprehensive metrics collection and monitoring system
    """
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self.collection_interval = 30  # seconds
        
        # Initialize core metrics
        self._initialize_metrics()
        logger.info("MetricsManager initialized")
    
    def _initialize_metrics(self):
        """Initialize core metrics"""
        
        # Performance metrics
        self.register_metric(
            "api_request_duration_seconds",
            MetricType.HISTOGRAM,
            "Duration of API requests in seconds"
        )
        
        self.register_metric(
            "job_processing_duration_seconds",
            MetricType.HISTOGRAM,
            "Duration of job processing in seconds"
        )
        
        self.register_metric(
            "pipeline_stage_duration_seconds",
            MetricType.HISTOGRAM,
            "Duration of individual pipeline stages in seconds"
        )
        
        # Business metrics
        self.register_metric(
            "jobs_created_total",
            MetricType.COUNTER,
            "Total number of jobs created"
        )
        
        self.register_metric(
            "jobs_completed_total",
            MetricType.COUNTER,
            "Total number of jobs completed successfully"
        )
        
        self.register_metric(
            "jobs_failed_total",
            MetricType.COUNTER,
            "Total number of jobs that failed"
        )
        
        self.register_metric(
            "webhook_deliveries_total",
            MetricType.COUNTER,
            "Total number of webhook deliveries attempted"
        )
        
        self.register_metric(
            "webhook_delivery_success_total",
            MetricType.COUNTER,
            "Total number of successful webhook deliveries"
        )
        
        # System metrics
        self.register_metric(
            "active_connections",
            MetricType.GAUGE,
            "Number of active SSE connections"
        )
        
        self.register_metric(
            "memory_usage_bytes",
            MetricType.GAUGE,
            "Memory usage in bytes"
        )
        
        self.register_metric(
            "cpu_usage_percent",
            MetricType.GAUGE,
            "CPU usage percentage"
        )
        
        self.register_metric(
            "cache_hit_rate_percent",
            MetricType.GAUGE,
            "Cache hit rate percentage"
        )

        self.register_metric(
            "cache_total_entries",
            MetricType.GAUGE,
            "Total number of cache entries"
        )

        self.register_metric(
            "cache_expired_entries",
            MetricType.GAUGE,
            "Number of expired cache entries"
        )

        self.register_metric(
            "process_memory_bytes",
            MetricType.GAUGE,
            "Process memory usage in bytes"
        )

        self.register_metric(
            "process_cpu_percent",
            MetricType.GAUGE,
            "Process CPU usage percentage"
        )

        # Error metrics
        self.register_metric(
            "errors_total",
            MetricType.COUNTER,
            "Total number of errors by type"
        )
        
        self.register_metric(
            "retry_attempts_total",
            MetricType.COUNTER,
            "Total number of retry attempts"
        )
    
    def register_metric(self, name: str, metric_type: MetricType, description: str, labels: Optional[Dict[str, str]] = None):
        """Register a new metric"""
        if name in self.metrics:
            logger.warning(f"Metric '{name}' already exists, skipping registration")
            return
        
        metric = Metric(
            name=name,
            metric_type=metric_type,
            description=description,
            labels=labels or {}
        )
        
        self.metrics[name] = metric
        logger.debug(f"Registered metric: {name} ({metric_type.value})")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        metric = self.metrics.get(name)
        if not metric:
            logger.warning(f"Metric '{name}' not found")
            return
        
        if metric.metric_type != MetricType.COUNTER:
            logger.warning(f"Metric '{name}' is not a counter")
            return
        
        current_value = metric.get_current_value() or 0
        metric.add_value(current_value + value, labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        metric = self.metrics.get(name)
        if not metric:
            logger.warning(f"Metric '{name}' not found")
            return
        
        if metric.metric_type != MetricType.GAUGE:
            logger.warning(f"Metric '{name}' is not a gauge")
            return
        
        metric.add_value(value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric value"""
        metric = self.metrics.get(name)
        if not metric:
            logger.warning(f"Metric '{name}' not found")
            return
        
        if metric.metric_type != MetricType.HISTOGRAM:
            logger.warning(f"Metric '{name}' is not a histogram")
            return
        
        metric.add_value(value, labels)
    
    def start_timer(self, name: str) -> 'TimerContext':
        """Start a timer for measuring duration"""
        return TimerContext(self, name)
    
    async def start_collection(self):
        """Start automatic metrics collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop automatic metrics collection"""
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Background task for collecting system metrics"""
        while self.is_collecting:
            try:
                await asyncio.sleep(self.collection_interval)
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect cache metrics
                await self._collect_cache_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.set_gauge("memory_usage_bytes", memory_info.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("cpu_usage_percent", cpu_percent)
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss
            self.set_gauge("process_memory_bytes", process_memory)
            
            process_cpu = process.cpu_percent()
            self.set_gauge("process_cpu_percent", process_cpu)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    async def _collect_cache_metrics(self):
        """Collect cache performance metrics"""
        try:
            from .cache_manager import cache_manager
            
            stats = cache_manager.get_stats()
            self.set_gauge("cache_hit_rate_percent", stats.get("hit_rate_percentage", 0))
            self.set_gauge("cache_total_entries", stats.get("total_entries", 0))
            self.set_gauge("cache_expired_entries", stats.get("expired_entries", 0))
            
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {str(e)}")
    
    def get_metric_summary(self, name: str, minutes: int = 5) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric"""
        metric = self.metrics.get(name)
        if not metric:
            return None
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_values = [
            v.value for v in metric.values
            if v.timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return {
                "name": name,
                "type": metric.metric_type.value,
                "description": metric.description,
                "count": 0,
                "current_value": metric.get_current_value()
            }
        
        summary = {
            "name": name,
            "type": metric.metric_type.value,
            "description": metric.description,
            "count": len(recent_values),
            "current_value": metric.get_current_value(),
            "min": min(recent_values),
            "max": max(recent_values),
            "average": sum(recent_values) / len(recent_values)
        }
        
        # Add percentiles for histograms
        if metric.metric_type == MetricType.HISTOGRAM:
            sorted_values = sorted(recent_values)
            count = len(sorted_values)
            
            summary.update({
                "p50": sorted_values[int(count * 0.5)] if count > 0 else 0,
                "p90": sorted_values[int(count * 0.9)] if count > 0 else 0,
                "p95": sorted_values[int(count * 0.95)] if count > 0 else 0,
                "p99": sorted_values[int(count * 0.99)] if count > 0 else 0
            })
        
        return summary
    
    def get_all_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get summary for all metrics"""
        summaries = {}
        
        for name in self.metrics:
            summary = self.get_metric_summary(name, minutes)
            if summary:
                summaries[name] = summary
        
        return {
            "metrics": summaries,
            "collection_interval_seconds": self.collection_interval,
            "total_metrics": len(self.metrics),
            "collection_active": self.is_collecting,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for metric in self.metrics.values():
            # Add metric description
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} {metric.metric_type.value}")
            
            # Add current value
            current_value = metric.get_current_value()
            if current_value is not None:
                labels_str = ""
                if metric.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f"{metric.name}{labels_str} {current_value}")
            
            lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, metrics_manager: MetricsManager, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.metrics_manager = metrics_manager
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_manager.record_histogram(self.metric_name, duration, self.labels)


# Global metrics manager instance
metrics_manager = MetricsManager()


# Convenience functions for common metrics
def increment_jobs_created():
    """Increment jobs created counter"""
    metrics_manager.increment_counter("jobs_created_total")


def increment_jobs_completed():
    """Increment jobs completed counter"""
    metrics_manager.increment_counter("jobs_completed_total")


def increment_jobs_failed():
    """Increment jobs failed counter"""
    metrics_manager.increment_counter("jobs_failed_total")


def record_api_request_duration(duration: float, endpoint: str):
    """Record API request duration"""
    metrics_manager.record_histogram(
        "api_request_duration_seconds",
        duration,
        {"endpoint": endpoint}
    )


def record_job_processing_duration(duration: float, status: str):
    """Record job processing duration"""
    metrics_manager.record_histogram(
        "job_processing_duration_seconds",
        duration,
        {"status": status}
    )


def set_active_connections(count: int):
    """Set number of active SSE connections"""
    metrics_manager.set_gauge("active_connections", count)


async def start_metrics_collection():
    """Start the global metrics manager"""
    await metrics_manager.start_collection()


async def stop_metrics_collection():
    """Stop the global metrics manager"""
    await metrics_manager.stop_collection()
