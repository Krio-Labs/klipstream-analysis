#!/usr/bin/env python3
"""
Simple Health Check Script for CPU Cloud Run Deployment

This script checks the basic health of the CPU-only transcription service.
"""

import os
import sys
import json
import requests
from datetime import datetime

def check_basic_imports():
    """Check if basic imports work"""
    try:
        import fastapi
        import uvicorn
        import torch
        import transformers
        return {"available": True}
    except Exception as e:
        return {"available": False, "error": str(e)}

def check_api_import():
    """Check if API can be imported"""
    try:
        # Try to import the API module
        sys.path.insert(0, '/app')
        from api.main import app
        return {"available": True}
    except Exception as e:
        return {"available": False, "error": str(e)}

def check_port_8080():
    """Check if something is listening on port 8080"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        return {
            "available": True,
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

def main():
    """Main health check function"""
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "healthy",
        "basic_imports": check_basic_imports(),
        "api_import": check_api_import(),
        "port_8080": check_port_8080(),
        "environment": {
            "gpu_transcription_enabled": os.getenv("ENABLE_GPU_TRANSCRIPTION", "false").lower() == "true",
            "transcription_method": os.getenv("TRANSCRIPTION_METHOD", "deepgram"),
            "cloud_environment": os.getenv("CLOUD_ENVIRONMENT", "false").lower() == "true",
            "fastapi_mode": os.getenv("FASTAPI_MODE", "false").lower() == "true"
        }
    }
    
    # Determine overall health
    issues = []
    
    if not health_status["basic_imports"]["available"]:
        issues.append("Basic imports failed")
    
    if not health_status["api_import"]["available"]:
        issues.append("API import failed")
    
    # Port 8080 check is optional for health check script
    # (it might not be running yet when health check runs)
    
    if issues:
        health_status["status"] = "degraded"
        health_status["issues"] = issues
        health_status["message"] = "; ".join(issues)
    
    print(json.dumps(health_status, indent=2))
    return 0 if health_status["status"] == "healthy" else 1

if __name__ == "__main__":
    sys.exit(main())
