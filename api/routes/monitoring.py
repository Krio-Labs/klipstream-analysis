"""
Monitoring and Analytics API Routes

This module provides API endpoints for system monitoring, alerts, and analytics.
"""

from fastapi import APIRouter, HTTPException, Request, Depends, Query
from typing import Dict, Any, Optional, List
from datetime import datetime

from api.services.monitoring_manager import AlertLevel
from utils.logging_setup import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


def get_monitoring_manager(request: Request):
    """Dependency to get monitoring manager from app state"""
    if not hasattr(request.app.state, 'monitoring_manager'):
        raise HTTPException(status_code=503, detail="Monitoring manager not available")
    return request.app.state.monitoring_manager


@router.get("/metrics")
async def get_metrics_summary(monitoring_manager = Depends(get_monitoring_manager)) -> Dict[str, Any]:
    """
    Get comprehensive system metrics summary
    
    Returns:
        Current system metrics and performance data
    """
    try:
        summary = monitoring_manager.get_metrics_summary()
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": summary
        }
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/health")
async def get_system_health(monitoring_manager = Depends(get_monitoring_manager)) -> Dict[str, Any]:
    """
    Get overall system health status
    
    Returns:
        System health assessment with status and recommendations
    """
    try:
        health = monitoring_manager.get_health_status()
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "health": health
        }
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")


@router.get("/alerts")
async def get_alerts(
    level: Optional[str] = Query(None, description="Filter by alert level (info, warning, error, critical)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts to return"),
    monitoring_manager = Depends(get_monitoring_manager)
) -> Dict[str, Any]:
    """
    Get recent system alerts
    
    Args:
        level: Optional alert level filter
        limit: Maximum number of alerts to return
        
    Returns:
        List of recent alerts with details
    """
    try:
        alert_level = None
        if level:
            try:
                alert_level = AlertLevel(level.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid alert level. Must be one of: {[l.value for l in AlertLevel]}"
                )
        
        alerts = monitoring_manager.get_alerts(level=alert_level, limit=limit)
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "alerts": alerts,
            "total_count": len(alerts),
            "filter": {
                "level": level,
                "limit": limit
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    monitoring_manager = Depends(get_monitoring_manager)
) -> Dict[str, Any]:
    """
    Mark an alert as resolved
    
    Args:
        alert_id: ID of the alert to resolve
        
    Returns:
        Success confirmation
    """
    try:
        success = await monitoring_manager.resolve_alert(alert_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Alert {alert_id} resolved successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")


@router.post("/metrics/custom")
async def add_custom_metric(
    metric_data: Dict[str, Any],
    monitoring_manager = Depends(get_monitoring_manager)
) -> Dict[str, Any]:
    """
    Add a custom application metric
    
    Args:
        metric_data: Dictionary containing metric name, value, and optional labels
        
    Returns:
        Success confirmation
    """
    try:
        # Validate required fields
        if "name" not in metric_data or "value" not in metric_data:
            raise HTTPException(
                status_code=400, 
                detail="Metric data must contain 'name' and 'value' fields"
            )
        
        name = metric_data["name"]
        value = float(metric_data["value"])
        labels = metric_data.get("labels", {})
        
        monitoring_manager.add_custom_metric(name, value, labels)
        
        return {
            "status": "success",
            "message": f"Custom metric '{name}' added successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metric value: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding custom metric: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add custom metric: {str(e)}")


@router.get("/performance")
async def get_performance_analytics(monitoring_manager = Depends(get_monitoring_manager)) -> Dict[str, Any]:
    """
    Get performance analytics and trends
    
    Returns:
        Performance metrics and analysis
    """
    try:
        summary = monitoring_manager.get_metrics_summary()
        
        # Extract performance insights
        performance_data = {
            "response_times": summary.get("request_stats", {}),
            "endpoint_performance": summary.get("endpoint_stats", {}),
            "system_resources": {
                "cpu_percent": summary.get("cpu_percent", {}),
                "memory_percent": summary.get("memory_percent", {}),
                "disk_percent": summary.get("disk_percent", {})
            }
        }
        
        # Calculate performance score (0-100)
        performance_score = 100
        
        # Deduct points for high resource usage
        if "current" in summary.get("cpu_percent", {}):
            cpu_usage = summary["cpu_percent"]["current"]
            if cpu_usage > 80:
                performance_score -= min(20, (cpu_usage - 80) * 2)
        
        if "current" in summary.get("memory_percent", {}):
            memory_usage = summary["memory_percent"]["current"]
            if memory_usage > 85:
                performance_score -= min(20, (memory_usage - 85) * 2)
        
        # Deduct points for slow response times
        request_stats = summary.get("request_stats", {})
        if "p95_response_time" in request_stats:
            p95_time = request_stats["p95_response_time"]
            if p95_time > 2.0:  # 2 seconds
                performance_score -= min(30, (p95_time - 2.0) * 10)
        
        performance_score = max(0, performance_score)
        
        # Determine performance grade
        if performance_score >= 90:
            grade = "A"
        elif performance_score >= 80:
            grade = "B"
        elif performance_score >= 70:
            grade = "C"
        elif performance_score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "performance": {
                "score": round(performance_score, 1),
                "grade": grade,
                "data": performance_data,
                "recommendations": _get_performance_recommendations(summary)
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance analytics: {str(e)}")


def _get_performance_recommendations(summary: Dict) -> List[str]:
    """Generate performance recommendations based on metrics"""
    recommendations = []
    
    # CPU recommendations
    if "current" in summary.get("cpu_percent", {}):
        cpu_usage = summary["cpu_percent"]["current"]
        if cpu_usage > 80:
            recommendations.append("High CPU usage detected. Consider scaling up or optimizing CPU-intensive operations.")
    
    # Memory recommendations
    if "current" in summary.get("memory_percent", {}):
        memory_usage = summary["memory_percent"]["current"]
        if memory_usage > 85:
            recommendations.append("High memory usage detected. Consider increasing memory allocation or optimizing memory usage.")
    
    # Response time recommendations
    request_stats = summary.get("request_stats", {})
    if "p95_response_time" in request_stats:
        p95_time = request_stats["p95_response_time"]
        if p95_time > 2.0:
            recommendations.append("Slow response times detected. Consider optimizing database queries or adding caching.")
    
    # Error rate recommendations
    endpoint_stats = summary.get("endpoint_stats", {})
    for endpoint, stats in endpoint_stats.items():
        if stats.get("error_rate", 0) > 0.05:  # 5%
            recommendations.append(f"High error rate on {endpoint}. Review error logs and implement fixes.")
    
    if not recommendations:
        recommendations.append("System performance is optimal. No immediate recommendations.")
    
    return recommendations


@router.get("/dashboard")
async def get_monitoring_dashboard(monitoring_manager = Depends(get_monitoring_manager)) -> Dict[str, Any]:
    """
    Get comprehensive monitoring dashboard data
    
    Returns:
        All monitoring data formatted for dashboard display
    """
    try:
        # Get all monitoring data
        metrics = monitoring_manager.get_metrics_summary()
        health = monitoring_manager.get_health_status()
        alerts = monitoring_manager.get_alerts(limit=10)
        
        # Count alerts by level
        alert_counts = {"info": 0, "warning": 0, "error": 0, "critical": 0}
        for alert in alerts:
            level = alert.get("level", "info")
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "dashboard": {
                "health": health,
                "metrics": metrics,
                "alerts": {
                    "recent": alerts[:5],  # Last 5 alerts
                    "counts": alert_counts,
                    "total": len(alerts)
                },
                "summary": {
                    "system_status": health.get("status", "unknown"),
                    "active_alerts": len([a for a in alerts if not a.get("resolved", False)]),
                    "performance_score": _calculate_simple_performance_score(metrics)
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


def _calculate_simple_performance_score(metrics: Dict) -> float:
    """Calculate a simple performance score from metrics"""
    score = 100.0
    
    # Deduct for high CPU
    if "current" in metrics.get("cpu_percent", {}):
        cpu = metrics["cpu_percent"]["current"]
        if cpu > 80:
            score -= (cpu - 80) * 0.5
    
    # Deduct for high memory
    if "current" in metrics.get("memory_percent", {}):
        memory = metrics["memory_percent"]["current"]
        if memory > 85:
            score -= (memory - 85) * 0.5
    
    return max(0, round(score, 1))
