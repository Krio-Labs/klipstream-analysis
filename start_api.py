#!/usr/bin/env python3
"""
Startup script for the KlipStream Analysis API

This script handles the startup of the FastAPI application with proper error handling.
"""

import os
import sys
import logging
import traceback

# Add the app directory to Python path
sys.path.insert(0, '/app')
sys.path.insert(0, os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test critical imports before starting the server"""
    try:
        logger.info("Testing critical imports...")
        
        import fastapi
        logger.info("✅ FastAPI imported")
        
        import uvicorn
        logger.info("✅ Uvicorn imported")
        
        # Test API imports
        from api.main import app
        logger.info("✅ API main module imported")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def start_server():
    """Start the FastAPI server"""
    try:
        logger.info("🚀 Starting KlipStream Analysis API...")
        
        # Test imports first
        if not test_imports():
            logger.error("❌ Import tests failed, cannot start server")
            return False
        
        # Import the app
        from api.main import app
        import uvicorn
        
        # Get port from environment
        port = int(os.getenv("PORT", 8080))
        
        logger.info(f"🌐 Starting server on 0.0.0.0:{port}")
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level="info",
            access_log=True
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def fallback_server():
    """Start a minimal fallback server if main API fails"""
    try:
        logger.info("🔄 Starting fallback server...")
        
        from fastapi import FastAPI
        import uvicorn
        
        fallback_app = FastAPI(title="KlipStream Fallback API")
        
        @fallback_app.get("/")
        async def root():
            return {
                "status": "fallback",
                "message": "Main API failed to start, running fallback server",
                "version": "2.0.0-fallback"
            }
        
        @fallback_app.get("/health")
        async def health():
            return {"status": "degraded", "mode": "fallback"}
        
        port = int(os.getenv("PORT", 8080))
        logger.info(f"🌐 Starting fallback server on 0.0.0.0:{port}")
        
        uvicorn.run(
            fallback_app,
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level="info"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback server failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🎬 KlipStream Analysis API Startup")
    logger.info("=" * 50)
    
    # Try to start the main server
    if not start_server():
        logger.warning("⚠️  Main server failed, trying fallback...")
        
        # Try fallback server
        if not fallback_server():
            logger.error("❌ Both main and fallback servers failed")
            sys.exit(1)
    
    logger.info("✅ Server started successfully")
