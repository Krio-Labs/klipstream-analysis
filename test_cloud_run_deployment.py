#!/usr/bin/env python3
"""
Cloud Run Deployment Test Suite

This script tests the deployed Cloud Run service to ensure all endpoints
are working correctly and the service is ready for production use.
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Cloud Run service URL
BASE_URL = "https://klipstream-analysis-771064872704.us-central1.run.app"

def test_health_check() -> bool:
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_root_endpoint() -> bool:
    """Test the root endpoint"""
    print("🔍 Testing root endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint passed: {data.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Root endpoint error: {str(e)}")
        return False

def test_minimal_endpoint() -> bool:
    """Test the minimal test endpoint"""
    print("🔍 Testing minimal test endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/test", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Minimal test endpoint passed: {data.get('message', 'Unknown')}")
            return True
        else:
            print(f"❌ Minimal test endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Minimal test endpoint error: {str(e)}")
        return False

def run_all_tests() -> Dict[str, bool]:
    """Run all tests and return results"""
    print("🚀 Starting Cloud Run Deployment Tests")
    print("=" * 50)
    
    tests = {
        "Health Check": test_health_check,
        "Root Endpoint": test_root_endpoint,
        "Minimal Test Endpoint": test_minimal_endpoint,
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        print(f"\n📋 Running: {test_name}")
        results[test_name] = test_func()
        time.sleep(1)  # Brief pause between tests
    
    return results

def print_summary(results: Dict[str, bool]) -> None:
    """Print test summary"""
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Cloud Run deployment is successful.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    print(f"Testing Cloud Run service at: {BASE_URL}")
    print(f"Test started at: {datetime.now().isoformat()}")
    
    results = run_all_tests()
    success = print_summary(results)
    
    sys.exit(0 if success else 1)
