#!/usr/bin/env python3
"""
Health Check Script for Cloud Run GPU Deployment

This script checks the health of the GPU-enabled transcription service.
"""

import os
import sys
import json
from datetime import datetime

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
            return {
                "available": True,
                "count": gpu_count,
                "name": gpu_name,
                "memory_gb": round(memory_total, 1)
            }
        else:
            return {"available": False}
    except Exception as e:
        return {"available": False, "error": str(e)}

def check_nemo():
    """Check NeMo availability"""
    try:
        import nemo.collections.asr as nemo_asr
        return {"available": True}
    except Exception as e:
        return {"available": False, "error": str(e)}

def check_api():
    """Check API availability"""
    try:
        from api.main import app
        return {"available": True}
    except Exception as e:
        return {"available": False, "error": str(e)}

def main():
    """Main health check function"""
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "healthy",
        "gpu": check_gpu(),
        "nemo": check_nemo(),
        "api": check_api(),
        "environment": {
            "gpu_transcription_enabled": os.getenv("ENABLE_GPU_TRANSCRIPTION", "false").lower() == "true",
            "transcription_method": os.getenv("TRANSCRIPTION_METHOD", "auto"),
            "cloud_environment": os.getenv("CLOUD_ENVIRONMENT", "false").lower() == "true"
        }
    }
    
    # Add CUDA version if available
    try:
        import torch
        if torch.cuda.is_available():
            health_status["environment"]["cuda_version"] = torch.version.cuda
            health_status["environment"]["pytorch_version"] = torch.__version__
    except:
        pass
    
    # Determine overall health
    issues = []
    
    if not health_status["api"]["available"]:
        issues.append("API not available")
    
    if health_status["environment"]["gpu_transcription_enabled"] and not health_status["gpu"]["available"]:
        issues.append("GPU transcription enabled but GPU not available")
    
    if not health_status["nemo"]["available"]:
        issues.append("NeMo not available")
    
    if issues:
        health_status["status"] = "degraded"
        health_status["issues"] = issues
        health_status["message"] = "; ".join(issues)
    
    print(json.dumps(health_status, indent=2))
    return 0 if health_status["status"] == "healthy" else 1

if __name__ == "__main__":
    sys.exit(main())
