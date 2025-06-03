"""
Enhanced Monitoring System for KlipStream Analysis API

This module provides comprehensive monitoring, alerting, and analytics
for the analysis pipeline and API performance.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict

import psutil

from utils.logging_setup import setup_logger

logger = setup_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Represents a system alert"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MetricPoint:
    """A single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float]


class MonitoringManager:
    """
    Enhanced monitoring system for comprehensive system and application monitoring
    
    Features:
    - Real-time system metrics collection
    - Custom application metrics
    - Alert management and notifications
    - Performance analytics and trends
    - Health check monitoring
    - Resource usage tracking
    """
    
    def __init__(self, 
                 collection_interval: int = 30,
                 retention_hours: int = 24,
                 alert_thresholds: Optional[Dict] = None):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        
        # Metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=int(retention_hours * 3600 / collection_interval)))
        self._custom_metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        
        # Alerts
        self._alerts: List[Alert] = []
        self._alert_callbacks: List[Callable] = []
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._last_network_stats = None
        
        # Performance tracking
        self._request_times: deque = deque(maxlen=1000)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._endpoint_stats: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        
        logger.info(f"MonitoringManager initialized with {collection_interval}s interval, {retention_hours}h retention")
    
    def _default_thresholds(self) -> Dict:
        """Default alert thresholds"""
        return {
            "cpu_percent": {"warning": 80, "critical": 95},
            "memory_percent": {"warning": 85, "critical": 95},
            "disk_percent": {"warning": 85, "critical": 95},
            "error_rate": {"warning": 0.05, "critical": 0.1},  # 5% and 10%
            "response_time_p95": {"warning": 5.0, "critical": 10.0},  # seconds
            "queue_length": {"warning": 50, "critical": 100},
            "failed_jobs_rate": {"warning": 0.1, "critical": 0.2}  # 10% and 20%
        }
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self._monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self._monitoring_active:
                await self._collect_system_metrics()
                await self._check_alerts()
                await self._cleanup_old_data()
                await asyncio.sleep(self.collection_interval)
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric("cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._add_metric("memory_percent", memory.percent)
            self._add_metric("memory_used_gb", memory.used / (1024**3))
            self._add_metric("memory_total_gb", memory.total / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self._add_metric("disk_percent", disk.percent)
            self._add_metric("disk_used_gb", disk.used / (1024**3))
            self._add_metric("disk_total_gb", disk.total / (1024**3))
            
            # Network metrics
            network = psutil.net_io_counters()
            if self._last_network_stats:
                bytes_sent_delta = network.bytes_sent - self._last_network_stats.bytes_sent
                bytes_recv_delta = network.bytes_recv - self._last_network_stats.bytes_recv
                self._add_metric("network_bytes_sent_rate", bytes_sent_delta / self.collection_interval)
                self._add_metric("network_bytes_recv_rate", bytes_recv_delta / self.collection_interval)
            self._last_network_stats = network
            
            # Process metrics
            process_count = len(psutil.pids())
            self._add_metric("process_count", process_count)
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                self._add_metric("load_average_1m", load_avg[0])
                self._add_metric("load_average_5m", load_avg[1])
                self._add_metric("load_average_15m", load_avg[2])
            except AttributeError:
                # getloadavg not available on Windows
                pass
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def _add_metric(self, name: str, value: float, labels: Optional[Dict] = None):
        """Add a metric data point"""
        timestamp = datetime.utcnow()
        self._metrics[name].append(MetricPoint(timestamp, value, labels or {}))
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        current_time = datetime.utcnow()
        
        # Check system metrics against thresholds
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name in self._metrics and self._metrics[metric_name]:
                latest_value = self._metrics[metric_name][-1].value
                
                # Check critical threshold
                if "critical" in thresholds and latest_value >= thresholds["critical"]:
                    await self._create_alert(
                        AlertLevel.CRITICAL,
                        f"Critical {metric_name}",
                        f"{metric_name} is at critical level: {latest_value:.2f}",
                        {"metric": metric_name, "value": latest_value, "threshold": thresholds["critical"]}
                    )
                
                # Check warning threshold
                elif "warning" in thresholds and latest_value >= thresholds["warning"]:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        f"High {metric_name}",
                        f"{metric_name} is above warning level: {latest_value:.2f}",
                        {"metric": metric_name, "value": latest_value, "threshold": thresholds["warning"]}
                    )
        
        # Check error rates
        await self._check_error_rates()
        
        # Check response times
        await self._check_response_times()
    
    async def _check_error_rates(self):
        """Check for high error rates"""
        if not self._endpoint_stats:
            return
        
        for endpoint, stats in self._endpoint_stats.items():
            if stats["count"] > 0:
                error_rate = stats["errors"] / stats["count"]
                
                if error_rate >= self.alert_thresholds["error_rate"]["critical"]:
                    await self._create_alert(
                        AlertLevel.CRITICAL,
                        f"Critical Error Rate - {endpoint}",
                        f"Error rate for {endpoint} is {error_rate:.2%}",
                        {"endpoint": endpoint, "error_rate": error_rate, "total_requests": stats["count"]}
                    )
                elif error_rate >= self.alert_thresholds["error_rate"]["warning"]:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        f"High Error Rate - {endpoint}",
                        f"Error rate for {endpoint} is {error_rate:.2%}",
                        {"endpoint": endpoint, "error_rate": error_rate, "total_requests": stats["count"]}
                    )
    
    async def _check_response_times(self):
        """Check for slow response times"""
        if len(self._request_times) < 20:  # Need enough samples
            return
        
        # Calculate 95th percentile
        sorted_times = sorted(self._request_times)
        p95_index = int(0.95 * len(sorted_times))
        p95_time = sorted_times[p95_index]
        
        if p95_time >= self.alert_thresholds["response_time_p95"]["critical"]:
            await self._create_alert(
                AlertLevel.CRITICAL,
                "Critical Response Time",
                f"95th percentile response time is {p95_time:.2f}s",
                {"p95_response_time": p95_time, "sample_size": len(self._request_times)}
            )
        elif p95_time >= self.alert_thresholds["response_time_p95"]["warning"]:
            await self._create_alert(
                AlertLevel.WARNING,
                "Slow Response Time",
                f"95th percentile response time is {p95_time:.2f}s",
                {"p95_response_time": p95_time, "sample_size": len(self._request_times)}
            )
    
    async def _create_alert(self, level: AlertLevel, title: str, message: str, metadata: Dict):
        """Create a new alert"""
        alert_id = f"{int(time.time())}_{len(self._alerts)}"
        alert = Alert(
            id=alert_id,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            source="monitoring_manager",
            metadata=metadata
        )
        
        self._alerts.append(alert)
        logger.log(
            40 if level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else 30,
            f"ALERT [{level.value.upper()}] {title}: {message}"
        )
        
        # Notify alert callbacks
        for callback in self._alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        # Clean up custom metrics
        for metric_name in list(self._custom_metrics.keys()):
            self._custom_metrics[metric_name] = [
                point for point in self._custom_metrics[metric_name]
                if point.timestamp > cutoff_time
            ]
            if not self._custom_metrics[metric_name]:
                del self._custom_metrics[metric_name]
        
        # Clean up old alerts (keep for 7 days)
        alert_cutoff = datetime.utcnow() - timedelta(days=7)
        self._alerts = [alert for alert in self._alerts if alert.timestamp > alert_cutoff]
    
    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """Record API request metrics"""
        self._request_times.append(duration)
        
        stats = self._endpoint_stats[endpoint]
        stats["count"] += 1
        stats["total_time"] += duration
        
        if not success:
            stats["errors"] += 1
    
    def add_custom_metric(self, name: str, value: float, labels: Optional[Dict] = None):
        """Add a custom application metric"""
        point = MetricPoint(datetime.utcnow(), value, labels or {})
        self._custom_metrics[name].append(point)
    
    def get_metrics_summary(self) -> Dict:
        """Get a summary of current metrics"""
        summary = {}
        
        # Latest system metrics
        for metric_name, points in self._metrics.items():
            if points:
                latest = points[-1]
                summary[metric_name] = {
                    "current": latest.value,
                    "timestamp": latest.timestamp.isoformat()
                }
        
        # Request statistics
        if self._request_times:
            sorted_times = sorted(self._request_times)
            summary["request_stats"] = {
                "count": len(self._request_times),
                "avg_response_time": sum(self._request_times) / len(self._request_times),
                "p50_response_time": sorted_times[int(0.5 * len(sorted_times))],
                "p95_response_time": sorted_times[int(0.95 * len(sorted_times))],
                "p99_response_time": sorted_times[int(0.99 * len(sorted_times))]
            }
        
        # Endpoint statistics
        summary["endpoint_stats"] = {}
        for endpoint, stats in self._endpoint_stats.items():
            if stats["count"] > 0:
                summary["endpoint_stats"][endpoint] = {
                    "requests": stats["count"],
                    "avg_response_time": stats["total_time"] / stats["count"],
                    "error_rate": stats["errors"] / stats["count"],
                    "errors": stats["errors"]
                }
        
        return summary
    
    def get_alerts(self, level: Optional[AlertLevel] = None, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        alerts = self._alerts
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        # Sort by timestamp (newest first) and limit
        alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                "id": alert.id,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "metadata": alert.metadata,
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in alerts
        ]
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback function for alert notifications"""
        self._alert_callbacks.append(callback)
        logger.info("Alert callback added")
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                logger.info(f"Alert {alert_id} resolved")
                return True
        
        logger.warning(f"Alert {alert_id} not found")
        return False
    
    def get_health_status(self) -> Dict:
        """Get overall system health status"""
        if not self._metrics:
            return {"status": "unknown", "message": "No metrics available"}
        
        # Check critical metrics
        critical_issues = []
        warnings = []
        
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name in self._metrics and self._metrics[metric_name]:
                latest_value = self._metrics[metric_name][-1].value
                
                if "critical" in thresholds and latest_value >= thresholds["critical"]:
                    critical_issues.append(f"{metric_name}: {latest_value:.2f}")
                elif "warning" in thresholds and latest_value >= thresholds["warning"]:
                    warnings.append(f"{metric_name}: {latest_value:.2f}")
        
        if critical_issues:
            return {
                "status": "critical",
                "message": f"Critical issues detected: {', '.join(critical_issues)}",
                "issues": critical_issues,
                "warnings": warnings
            }
        elif warnings:
            return {
                "status": "warning",
                "message": f"Warnings detected: {', '.join(warnings)}",
                "warnings": warnings
            }
        else:
            return {
                "status": "healthy",
                "message": "All systems operating normally"
            }
