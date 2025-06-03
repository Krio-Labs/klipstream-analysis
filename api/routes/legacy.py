"""
Legacy API Routes for Backward Compatibility

This module provides backward compatibility with the old functions-framework endpoints
during the migration period. It wraps the old functionality in FastAPI endpoints.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from utils.logging_setup import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/legacy", tags=["legacy"])


@router.post("/run_pipeline")
async def legacy_run_pipeline(
    request: Request,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Legacy endpoint that mimics the old functions-framework run_pipeline function
    
    This endpoint provides backward compatibility for existing clients while
    internally using the new async job system.
    
    Args:
        request: FastAPI request object containing the video URL
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        Response in the old format for compatibility
    """
    try:
        # Parse request body
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Request body is required")
        
        try:
            request_data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
        
        # Extract video URL from request
        video_url = request_data.get('url')
        if not video_url:
            raise HTTPException(status_code=400, detail="Video URL is required")
        
        logger.info(f"Legacy endpoint called with URL: {video_url}")
        
        # For backward compatibility, we'll run the pipeline synchronously
        # but with a timeout to prevent Cloud Run timeouts
        try:
            # Import the original pipeline function
            from main import run_integrated_pipeline
            
            # Run the pipeline with the original logic
            result = await asyncio.wait_for(
                asyncio.to_thread(run_integrated_pipeline, video_url),
                timeout=3300  # 55 minutes (Cloud Run max is 60 minutes)
            )
            
            # Return result in the original format
            return {
                "status": "success",
                "message": "Analysis completed successfully",
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Pipeline timeout for URL: {video_url}")
            raise HTTPException(
                status_code=504,
                detail="Pipeline execution timed out. Please try again or use the new async API."
            )
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline execution failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in legacy endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/list_files")
async def legacy_list_files() -> Dict[str, Any]:
    """
    Legacy endpoint for listing files

    Returns:
        List of files in the output directory
    """
    try:
        # Simple file listing implementation
        import os

        output_dir = "/tmp/output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        files = []
        for root, dirs, filenames in os.walk(output_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, output_dir)
                files.append(relative_path)

        return {
            "status": "success",
            "files": files,
            "count": len(files),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.get("/list_output_files")
async def legacy_list_output_files() -> Dict[str, Any]:
    """
    Legacy endpoint for listing output files

    Returns:
        List of output files
    """
    try:
        # Simple output file listing implementation
        import os

        output_dirs = ["/tmp/output", "/tmp/output/Analysis", "/tmp/output/raw"]
        all_files = []

        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                for root, _, filenames in os.walk(output_dir):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(file_path, "/tmp/output")
                        all_files.append(relative_path)

        return {
            "status": "success",
            "files": all_files,
            "count": len(all_files),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing output files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list output files: {str(e)}")


@router.get("/migration_status")
async def get_migration_status() -> Dict[str, Any]:
    """
    Get information about the migration status and available endpoints
    
    Returns:
        Migration status and endpoint information
    """
    return {
        "status": "success",
        "migration": {
            "phase": "4",
            "description": "Legacy endpoints available for backward compatibility",
            "legacy_endpoints": [
                "/legacy/run_pipeline",
                "/legacy/list_files", 
                "/legacy/list_output_files"
            ],
            "new_endpoints": [
                "/api/v1/analysis",
                "/api/v1/status/{job_id}",
                "/api/v1/queue/status",
                "/api/v1/monitoring/health"
            ],
            "recommendation": "Please migrate to the new async API endpoints for better performance and features"
        },
        "compatibility": {
            "legacy_support": True,
            "deprecation_notice": "Legacy endpoints will be removed in Phase 5",
            "migration_guide": "/docs#migration-guide"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health")
async def legacy_health_check() -> Dict[str, Any]:
    """
    Legacy health check endpoint
    
    Returns:
        Health status in legacy format
    """
    try:
        # Check if the main pipeline components are available
        health_status = "healthy"
        components = {}
        
        # Check if required directories exist
        required_dirs = ["/tmp/output", "/tmp/downloads", "/tmp/data"]
        for dir_path in required_dirs:
            components[f"directory_{dir_path.replace('/', '_')}"] = os.path.exists(dir_path)
        
        # Check if model files exist
        model_files = [
            "/app/analysis_pipeline/chat/models/emotion_classifier_pipe_lr.pkl",
            "/app/analysis_pipeline/chat/models/highlight_classifier_pipe_lr.pkl"
        ]
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('.pkl', '')
            components[f"model_{model_name}"] = os.path.exists(model_file) and os.path.getsize(model_file) > 0
        
        # Check if any critical components are missing
        if not all(components.values()):
            health_status = "degraded"
        
        return {
            "status": health_status,
            "components": components,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "legacy",
            "migration_available": True
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
