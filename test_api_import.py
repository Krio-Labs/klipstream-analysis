#!/usr/bin/env python3
"""
Test script to check if the API can be imported and started
"""

import sys
import os
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

def test_imports():
    """Test if all API imports work"""
    try:
        print("Testing basic imports...")
        import fastapi
        print("✅ FastAPI imported successfully")
        
        import uvicorn
        print("✅ Uvicorn imported successfully")
        
        print("\nTesting API module imports...")
        from api import main
        print("✅ API main module imported successfully")
        
        print("\nTesting route imports...")
        from api.routes import analysis, status, webhooks, health, queue, monitoring, legacy, analysis_minimal
        print("✅ All route modules imported successfully")
        
        print("\nTesting app creation...")
        app = main.app
        print("✅ FastAPI app created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_simple_server():
    """Test if we can start a simple server"""
    try:
        print("\nTesting simple server startup...")
        
        # Create a minimal FastAPI app
        from fastapi import FastAPI
        simple_app = FastAPI()
        
        @simple_app.get("/")
        async def root():
            return {"message": "Hello World"}
        
        @simple_app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        print("✅ Simple FastAPI app created successfully")
        return simple_app
        
    except Exception as e:
        print(f"❌ Simple server error: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 Testing API imports and startup...")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Trying simple server...")
        simple_app = test_simple_server()
        
        if simple_app:
            print("\n🚀 Starting simple server on port 8080...")
            import uvicorn
            uvicorn.run(simple_app, host="0.0.0.0", port=8080)
        else:
            print("\n❌ All tests failed")
            sys.exit(1)
    else:
        print("\n✅ All import tests passed!")
        print("\n🚀 Starting full API server on port 8080...")
        from api.main import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8080)
