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
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from dotenv import load_dotenv

# Import routes
from .routes import analysis, status, webhooks, health, queue, monitoring, legacy, analysis_minimal

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
        # TEMPORARILY DISABLED: Background services causing event loop blocking
        logger.info("⚠️  Background services temporarily disabled for debugging")
        logger.info("🔧 This resolves the 17-minute request delay issue")

        # TODO: Re-enable with proper async handling
        # # Start cache manager
        # from .services.cache_manager import start_cache_manager
        # await start_cache_manager()
        # logger.info("✅ Cache manager started")

        # # Start metrics collection
        # from .services.metrics_manager import start_metrics_collection
        # await start_metrics_collection()
        # logger.info("✅ Metrics collection started")

        # # Start queue manager
        # from .services.queue_manager import QueueManager
        # app.state.queue_manager = QueueManager(max_concurrent_jobs=3, max_queue_size=100)
        # logger.info("✅ Queue manager initialized")

        # # Start monitoring manager
        # from .services.monitoring_manager import MonitoringManager
        # app.state.monitoring_manager = MonitoringManager(collection_interval=30, retention_hours=24)
        # await app.state.monitoring_manager.start_monitoring()
        # logger.info("✅ Monitoring manager started")

        logger.info("🎉 Minimal API started successfully (background services disabled)")

    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down KlipStream Analysis API")

    try:
        # TEMPORARILY DISABLED: Background services shutdown
        logger.info("⚠️  Background services shutdown disabled (services were not started)")

        # TODO: Re-enable when background services are re-enabled
        # # Stop monitoring manager
        # if hasattr(app.state, 'monitoring_manager'):
        #     await app.state.monitoring_manager.stop_monitoring()
        #     logger.info("✅ Monitoring manager stopped")

        # # Stop queue manager
        # if hasattr(app.state, 'queue_manager'):
        #     await app.state.queue_manager.stop_queue()
        #     logger.info("✅ Queue manager stopped")

        # # Stop metrics collection
        # from .services.metrics_manager import stop_metrics_collection
        # await stop_metrics_collection()
        # logger.info("✅ Metrics collection stopped")

        # # Stop cache manager
        # from .services.cache_manager import stop_cache_manager
        # await stop_cache_manager()
        # logger.info("✅ Cache manager stopped")

        logger.info("🎉 Minimal API shutdown completed")

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


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle FastAPI request validation errors with detailed information
    """
    logger.error(f"Request validation error for {request.method} {request.url}: {str(exc)}")
    try:
        body = await request.body()
        logger.error(f"Request body: {body.decode('utf-8') if body else 'Empty'}")
    except Exception as e:
        logger.error(f"Could not read request body: {str(e)}")

    return JSONResponse(
        status_code=422,
        content={
            "status": "validation_error",
            "message": "Request validation failed",
            "errors": exc.errors(),
            "timestamp": "2024-01-15T10:30:00Z",
            "request_info": {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers)
            }
        }
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    Handle Pydantic validation errors with detailed information
    """
    logger.error(f"Pydantic validation error for {request.method} {request.url}: {str(exc)}")
    try:
        body = await request.body()
        logger.error(f"Request body: {body.decode('utf-8') if body else 'Empty'}")
    except Exception as e:
        logger.error(f"Could not read request body: {str(e)}")

    return JSONResponse(
        status_code=422,
        content={
            "status": "validation_error",
            "message": "Pydantic validation failed",
            "errors": exc.errors(),
            "timestamp": "2024-01-15T10:30:00Z"
        }
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
app.include_router(analysis_minimal.router, prefix="/api/v1", tags=["analysis-minimal"])
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
