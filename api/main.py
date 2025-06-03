"""
FastAPI Application Main Module

This module sets up the FastAPI application with all routes, middleware, and configuration.
It serves as the entry point for the new asynchronous API.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Import routes
from .routes import analysis, status, webhooks, health, queue, monitoring, legacy

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting KlipStream Analysis API v2.0")
    logger.info("📊 Async job processing enabled")
    logger.info("🔄 Real-time status updates available")

    try:
        # Start cache manager
        from .services.cache_manager import start_cache_manager
        await start_cache_manager()
        logger.info("✅ Cache manager started")

        # Start metrics collection
        from .services.metrics_manager import start_metrics_collection
        await start_metrics_collection()
        logger.info("✅ Metrics collection started")

        # Start queue manager
        from .services.queue_manager import QueueManager
        app.state.queue_manager = QueueManager(max_concurrent_jobs=3, max_queue_size=100)
        logger.info("✅ Queue manager initialized")

        # Start monitoring manager
        from .services.monitoring_manager import MonitoringManager
        app.state.monitoring_manager = MonitoringManager(collection_interval=30, retention_hours=24)
        await app.state.monitoring_manager.start_monitoring()
        logger.info("✅ Monitoring manager started")

        logger.info("🎉 All Phase 3 services started successfully")

    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down KlipStream Analysis API")

    try:
        # Stop monitoring manager
        if hasattr(app.state, 'monitoring_manager'):
            await app.state.monitoring_manager.stop_monitoring()
            logger.info("✅ Monitoring manager stopped")

        # Stop queue manager
        if hasattr(app.state, 'queue_manager'):
            await app.state.queue_manager.stop_queue()
            logger.info("✅ Queue manager stopped")

        # Stop metrics collection
        from .services.metrics_manager import stop_metrics_collection
        await stop_metrics_collection()
        logger.info("✅ Metrics collection stopped")

        # Stop cache manager
        from .services.cache_manager import stop_cache_manager
        await stop_cache_manager()
        logger.info("✅ Cache manager stopped")

        logger.info("🎉 All services stopped successfully")

    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title="KlipStream Analysis API",
    description="Asynchronous video analysis service for Twitch VODs with real-time progress tracking",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware with proper OPTIONS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "error": str(exc),
            "timestamp": "2024-01-15T10:30:00Z"  # Will be dynamic
        }
    )


@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "name": "KlipStream Analysis API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Asynchronous video processing",
            "Real-time progress tracking",
            "Comprehensive error handling",
            "Server-sent events support"
        ],
        "endpoints": {
            "analysis": "/api/v1/analysis",
            "status": "/api/v1/analysis/{job_id}/status",
            "stream": "/api/v1/analysis/{job_id}/stream",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00Z",  # Will be dynamic
        "version": "2.0.0"
    }


# Include routers
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])
app.include_router(webhooks.router)
app.include_router(health.router)
app.include_router(queue.router)
app.include_router(monitoring.router)
app.include_router(legacy.router)


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
