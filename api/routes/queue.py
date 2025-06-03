"""
Queue Management API Routes

This module provides API endpoints for managing the job queue system.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Dict, Any, Optional
from datetime import datetime

from utils.logging_setup import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1/queue", tags=["queue"])


def get_queue_manager(request: Request):
    """Dependency to get queue manager from app state"""
    if not hasattr(request.app.state, 'queue_manager'):
        raise HTTPException(status_code=503, detail="Queue manager not available")
    return request.app.state.queue_manager


@router.get("/status")
async def get_queue_status(queue_manager = Depends(get_queue_manager)) -> Dict[str, Any]:
    """
    Get current queue status and metrics
    
    Returns:
        Dict containing queue status, metrics, and job information
    """
    try:
        status = await queue_manager.get_queue_status()
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "queue": status
        }
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")


@router.post("/pause")
async def pause_queue(queue_manager = Depends(get_queue_manager)) -> Dict[str, Any]:
    """
    Pause the job queue (stop processing new jobs)
    
    Returns:
        Success confirmation
    """
    try:
        await queue_manager.pause_queue()
        return {
            "status": "success",
            "message": "Queue paused successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error pausing queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to pause queue: {str(e)}")


@router.post("/resume")
async def resume_queue(queue_manager = Depends(get_queue_manager)) -> Dict[str, Any]:
    """
    Resume the job queue
    
    Returns:
        Success confirmation
    """
    try:
        await queue_manager.resume_queue()
        return {
            "status": "success",
            "message": "Queue resumed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resuming queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resume queue: {str(e)}")


@router.post("/drain")
async def drain_queue(queue_manager = Depends(get_queue_manager)) -> Dict[str, Any]:
    """
    Drain the queue (finish existing jobs, don't accept new ones)
    
    Returns:
        Success confirmation
    """
    try:
        await queue_manager.drain_queue()
        return {
            "status": "success",
            "message": "Queue draining - finishing existing jobs",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error draining queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to drain queue: {str(e)}")


@router.delete("/jobs/{job_id}")
async def remove_job_from_queue(
    job_id: str,
    queue_manager = Depends(get_queue_manager)
) -> Dict[str, Any]:
    """
    Remove a specific job from the queue
    
    Args:
        job_id: ID of the job to remove
        
    Returns:
        Success confirmation or error if job not found
    """
    try:
        success = await queue_manager.remove_job(job_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Job {job_id} removed from queue",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found in queue")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing job {job_id} from queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to remove job: {str(e)}")


@router.get("/jobs/{job_id}/position")
async def get_job_position(
    job_id: str,
    queue_manager = Depends(get_queue_manager)
) -> Dict[str, Any]:
    """
    Get the position of a job in the queue
    
    Args:
        job_id: ID of the job to check
        
    Returns:
        Job position information
    """
    try:
        position = queue_manager.get_job_position(job_id)
        
        if position is not None:
            return {
                "status": "success",
                "job_id": job_id,
                "position": position,
                "message": f"Job {job_id} is at position {position} in queue",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Check if job is currently being processed
            queue_status = await queue_manager.get_queue_status()
            if job_id in queue_status.get("active_job_ids", []):
                return {
                    "status": "success",
                    "job_id": job_id,
                    "position": -1,
                    "message": f"Job {job_id} is currently being processed",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found in queue")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job position: {str(e)}")


@router.get("/metrics")
async def get_queue_metrics(queue_manager = Depends(get_queue_manager)) -> Dict[str, Any]:
    """
    Get detailed queue metrics and performance statistics
    
    Returns:
        Comprehensive queue metrics
    """
    try:
        status = await queue_manager.get_queue_status()
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "queue_length": status["queue_length"],
                "active_jobs": status["active_jobs"],
                "active_workers": status["active_workers"],
                "max_workers": status["max_workers"],
                "max_queue_size": status["max_queue_size"],
                "utilization_percent": round((status["active_workers"] / status["max_workers"]) * 100, 2),
                "queue_utilization_percent": round((status["queue_length"] / status["max_queue_size"]) * 100, 2),
                "performance": status["metrics"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting queue metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue metrics: {str(e)}")


@router.get("/health")
async def get_queue_health(queue_manager = Depends(get_queue_manager)) -> Dict[str, Any]:
    """
    Get queue health status
    
    Returns:
        Queue health information
    """
    try:
        status = await queue_manager.get_queue_status()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        # Check for potential issues
        if status["queue_length"] >= status["max_queue_size"] * 0.9:
            health_status = "warning"
            issues.append("Queue is nearly full")
        
        if status["active_workers"] == 0 and status["queue_length"] > 0:
            health_status = "critical"
            issues.append("No active workers but jobs are queued")
        
        if status["metrics"]["total_failed"] > status["metrics"]["total_processed"] * 0.1:
            health_status = "warning"
            issues.append("High failure rate detected")
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "health": {
                "status": health_status,
                "queue_status": status["status"],
                "issues": issues,
                "summary": {
                    "queue_length": status["queue_length"],
                    "active_workers": status["active_workers"],
                    "uptime_hours": round(status["metrics"]["uptime_seconds"] / 3600, 2),
                    "total_processed": status["metrics"]["total_processed"],
                    "success_rate": round(
                        (status["metrics"]["total_processed"] / 
                         max(status["metrics"]["total_processed"] + status["metrics"]["total_failed"], 1)) * 100, 2
                    )
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting queue health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue health: {str(e)}")
