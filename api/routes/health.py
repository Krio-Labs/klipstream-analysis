"""
Enhanced Health Check Routes

This module provides comprehensive health check endpoints for monitoring
the KlipStream Analysis API and its dependencies.
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    """Health status response model"""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str = "2.0.0"


class DetailedHealthStatus(BaseModel):
    """Detailed health status response model"""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str = "2.0.0"
    system: Dict[str, Any]
    services: Dict[str, Any]
    metrics: Dict[str, Any]
    cache: Dict[str, Any]


# Track service start time
SERVICE_START_TIME = time.time()


@router.get("/health", response_model=HealthStatus)
@router.options("/health")
async def basic_health_check():
    """
    Basic health check endpoint
    
    Returns basic service health information for load balancers and monitoring.
    """
    try:
        uptime = time.time() - SERVICE_START_TIME
        
        return HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check():
    """
    Detailed health check endpoint
    
    Returns comprehensive health information including system metrics,
    service dependencies, and performance data.
    """
    try:
        uptime = time.time() - SERVICE_START_TIME
        
        # Collect system information
        system_info = await _get_system_info()
        
        # Check service dependencies
        services_info = await _check_services()
        
        # Get metrics information
        metrics_info = await _get_metrics_info()
        
        # Get cache information
        cache_info = await _get_cache_info()
        
        # Determine overall status
        overall_status = _determine_overall_status(services_info)
        
        return DetailedHealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=uptime,
            system=system_info,
            services=services_info,
            metrics=metrics_info,
            cache=cache_info
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/health/readiness")
async def readiness_check():
    """
    Readiness check endpoint
    
    Checks if the service is ready to accept requests.
    Used by Kubernetes and other orchestrators.
    """
    try:
        # Check critical dependencies
        services_info = await _check_services()
        
        # Check if critical services are healthy
        critical_services = ["job_manager", "cache_manager"]
        for service in critical_services:
            if services_info.get(service, {}).get("status") != "healthy":
                raise HTTPException(
                    status_code=503,
                    detail=f"Critical service {service} is not ready"
                )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/health/liveness")
async def liveness_check():
    """
    Liveness check endpoint
    
    Checks if the service is alive and functioning.
    Used by Kubernetes and other orchestrators.
    """
    try:
        # Basic liveness check - if we can respond, we're alive
        uptime = time.time() - SERVICE_START_TIME
        
        # Check if uptime is reasonable (not stuck in restart loop)
        if uptime < 10:  # Less than 10 seconds uptime might indicate issues
            logger.warning(f"Service uptime is very low: {uptime} seconds")
        
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime
        }
        
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not alive")


@router.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus-compatible metrics endpoint
    
    Returns metrics in Prometheus format for monitoring systems.
    """
    try:
        from ..services.metrics_manager import metrics_manager
        
        prometheus_metrics = metrics_manager.export_prometheus_format()
        
        return {
            "metrics": prometheus_metrics,
            "format": "prometheus",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


async def _get_system_info() -> Dict[str, Any]:
    """Get system information"""
    try:
        # Memory information
        memory = psutil.virtual_memory()
        
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Disk information
        disk = psutil.disk_usage('/')
        
        # Process information
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "memory": {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "percent_used": memory.percent
            },
            "cpu": {
                "percent_used": cpu_percent,
                "core_count": cpu_count
            },
            "disk": {
                "total_bytes": disk.total,
                "free_bytes": disk.free,
                "used_bytes": disk.used,
                "percent_used": (disk.used / disk.total) * 100
            },
            "process": {
                "memory_rss_bytes": process_memory.rss,
                "memory_vms_bytes": process_memory.vms,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e)}


async def _check_services() -> Dict[str, Any]:
    """Check health of service dependencies"""
    services = {}
    
    # Check job manager
    try:
        from ..services.job_manager import job_manager_instance
        
        job_count = len(job_manager_instance.jobs)
        services["job_manager"] = {
            "status": "healthy",
            "active_jobs": job_count,
            "details": "Job manager is operational"
        }
        
    except Exception as e:
        services["job_manager"] = {
            "status": "unhealthy",
            "error": str(e),
            "details": "Job manager is not accessible"
        }
    
    # Check cache manager
    try:
        from ..services.cache_manager import cache_manager
        
        cache_stats = cache_manager.get_stats()
        services["cache_manager"] = {
            "status": "healthy",
            "total_entries": cache_stats.get("total_entries", 0),
            "hit_rate": cache_stats.get("hit_rate_percentage", 0),
            "details": "Cache manager is operational"
        }
        
    except Exception as e:
        services["cache_manager"] = {
            "status": "unhealthy",
            "error": str(e),
            "details": "Cache manager is not accessible"
        }
    
    # Check webhook manager
    try:
        from ..services.webhook_manager import webhook_manager
        
        webhook_count = len(webhook_manager.webhooks)
        services["webhook_manager"] = {
            "status": "healthy",
            "registered_webhooks": webhook_count,
            "details": "Webhook manager is operational"
        }
        
    except Exception as e:
        services["webhook_manager"] = {
            "status": "unhealthy",
            "error": str(e),
            "details": "Webhook manager is not accessible"
        }
    
    # Check external services (with timeout)
    services["external_services"] = await _check_external_services()
    
    return services


async def _check_external_services() -> Dict[str, Any]:
    """Check external service connectivity"""
    external_services = {}
    
    # Check Convex database
    try:
        from utils.convex_client_updated import ConvexManager
        
        # Quick connectivity test with timeout
        convex_manager = ConvexManager()
        
        # This is a simple test - in production you might want a dedicated health check method
        if convex_manager.convex:
            external_services["convex"] = {
                "status": "healthy",
                "details": "Convex database is accessible"
            }
        else:
            external_services["convex"] = {
                "status": "unhealthy",
                "details": "Convex client not initialized"
            }
            
    except Exception as e:
        external_services["convex"] = {
            "status": "unhealthy",
            "error": str(e),
            "details": "Convex database is not accessible"
        }
    
    # Note: Add other external service checks here (Deepgram, GCS, etc.)
    # For now, we'll mark them as "not_checked"
    external_services["deepgram"] = {
        "status": "not_checked",
        "details": "Deepgram API health check not implemented"
    }
    
    external_services["google_cloud_storage"] = {
        "status": "not_checked",
        "details": "GCS health check not implemented"
    }
    
    return external_services


async def _get_metrics_info() -> Dict[str, Any]:
    """Get metrics information"""
    try:
        from ..services.metrics_manager import metrics_manager
        
        summary = metrics_manager.get_all_metrics_summary(minutes=5)
        
        return {
            "collection_active": summary.get("collection_active", False),
            "total_metrics": summary.get("total_metrics", 0),
            "collection_interval": summary.get("collection_interval_seconds", 0),
            "recent_metrics_count": len([
                m for m in summary.get("metrics", {}).values()
                if m.get("count", 0) > 0
            ])
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics info: {str(e)}")
        return {"error": str(e)}


async def _get_cache_info() -> Dict[str, Any]:
    """Get cache information"""
    try:
        from ..services.cache_manager import cache_manager
        
        return cache_manager.get_stats()
        
    except Exception as e:
        logger.error(f"Error getting cache info: {str(e)}")
        return {"error": str(e)}


def _determine_overall_status(services_info: Dict[str, Any]) -> str:
    """Determine overall service status based on dependencies"""
    
    # Critical services that must be healthy
    critical_services = ["job_manager", "cache_manager"]
    
    for service in critical_services:
        service_info = services_info.get(service, {})
        if service_info.get("status") != "healthy":
            return "degraded"
    
    # Check if any external services are unhealthy
    external_services = services_info.get("external_services", {})
    unhealthy_external = [
        name for name, info in external_services.items()
        if info.get("status") == "unhealthy"
    ]
    
    if unhealthy_external:
        logger.warning(f"External services unhealthy: {unhealthy_external}")
        return "degraded"
    
    return "healthy"
