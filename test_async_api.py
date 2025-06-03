#!/usr/bin/env python3
"""
Test Script for Async API Implementation

This script tests the new FastAPI endpoints and job management system.
It verifies that Phase 1 implementation is working correctly.
"""

import asyncio
import json
import time
import requests
from datetime import datetime

# Test configuration
API_BASE_URL = "http://localhost:3000"
TEST_VIDEO_URL = "https://www.twitch.tv/videos/2434635255"


def test_api_health():
    """Test API health and basic endpoints"""
    print("🔍 Testing API health...")
    
    try:
        # Test root endpoint
        response = requests.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "KlipStream Analysis API"
        assert data["version"] == "2.0.0"
        print("✅ Root endpoint working")
        
        # Test health endpoint
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health endpoint working")
        
        return True
        
    except Exception as e:
        print(f"❌ API health test failed: {str(e)}")
        return False


def test_analysis_endpoint():
    """Test the analysis endpoint"""
    print("🔍 Testing analysis endpoint...")
    
    try:
        # Test with valid URL
        payload = {
            "url": TEST_VIDEO_URL,
            "callback_url": "https://example.com/webhook"
        }
        
        response = requests.post(f"{API_BASE_URL}/api/v1/analysis", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "job_id" in data
        assert data["video_id"] == "2434635255"
        assert data["progress"]["current_stage"] == "Queued"
        assert data["progress"]["percentage"] == 0.0
        
        job_id = data["job_id"]
        print(f"✅ Analysis started successfully with job ID: {job_id}")
        
        return job_id
        
    except Exception as e:
        print(f"❌ Analysis endpoint test failed: {str(e)}")
        return None


def test_invalid_url():
    """Test analysis endpoint with invalid URL"""
    print("🔍 Testing invalid URL handling...")
    
    try:
        payload = {"url": "https://invalid-url.com/video/123"}
        
        response = requests.post(f"{API_BASE_URL}/api/v1/analysis", json=payload)
        assert response.status_code == 400
        
        data = response.json()
        assert "error" in data["detail"]
        print("✅ Invalid URL properly rejected")
        
        return True
        
    except Exception as e:
        print(f"❌ Invalid URL test failed: {str(e)}")
        return False


def test_status_endpoint(job_id):
    """Test the status endpoint"""
    print(f"🔍 Testing status endpoint for job {job_id}...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/analysis/{job_id}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "progress" in data
        print("✅ Status endpoint working")
        
        return True
        
    except Exception as e:
        print(f"❌ Status endpoint test failed: {str(e)}")
        return False


def test_job_details_endpoint(job_id):
    """Test the job details endpoint"""
    print(f"🔍 Testing job details endpoint for job {job_id}...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/analysis/{job_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["job_id"] == job_id
        assert "progress" in data
        assert "metadata" in data
        print("✅ Job details endpoint working")
        
        return True
        
    except Exception as e:
        print(f"❌ Job details endpoint test failed: {str(e)}")
        return False


def test_list_jobs_endpoint():
    """Test the list jobs endpoint"""
    print("🔍 Testing list jobs endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "jobs" in data
        print(f"✅ List jobs endpoint working (found {len(data['jobs'])} jobs)")
        
        return True
        
    except Exception as e:
        print(f"❌ List jobs endpoint test failed: {str(e)}")
        return False


def test_sse_stream(job_id, duration=10):
    """Test Server-Sent Events stream"""
    print(f"🔍 Testing SSE stream for job {job_id} (for {duration} seconds)...")
    
    try:
        import sseclient  # You might need to install this: pip install sseclient-py
        
        response = requests.get(f"{API_BASE_URL}/api/v1/analysis/{job_id}/stream", stream=True)
        client = sseclient.SSEClient(response)
        
        start_time = time.time()
        event_count = 0
        
        for event in client.events():
            if time.time() - start_time > duration:
                break
                
            if event.data:
                try:
                    data = json.loads(event.data)
                    print(f"📡 SSE Event: {data.get('status', 'unknown')} - {data.get('message', 'no message')}")
                    event_count += 1
                except json.JSONDecodeError:
                    print(f"📡 SSE Event (raw): {event.data}")
                    event_count += 1
        
        print(f"✅ SSE stream working (received {event_count} events)")
        return True
        
    except ImportError:
        print("⚠️  SSE test skipped (sseclient-py not installed)")
        print("   Install with: pip install sseclient-py")
        return True
    except Exception as e:
        print(f"❌ SSE stream test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Async API Tests")
    print("=" * 50)
    
    # Check if API is running
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=5)
    except requests.exceptions.RequestException:
        print("❌ API is not running!")
        print("   Start the API with: python -m api.main")
        return False
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Test 1: API Health
    total_tests += 1
    if test_api_health():
        tests_passed += 1
    
    # Test 2: Analysis Endpoint
    total_tests += 1
    job_id = test_analysis_endpoint()
    if job_id:
        tests_passed += 1
        
        # Test 3: Status Endpoint (only if we have a job ID)
        total_tests += 1
        if test_status_endpoint(job_id):
            tests_passed += 1
        
        # Test 4: Job Details Endpoint
        total_tests += 1
        if test_job_details_endpoint(job_id):
            tests_passed += 1
        
        # Test 5: SSE Stream (short test)
        total_tests += 1
        if test_sse_stream(job_id, duration=5):
            tests_passed += 1
    
    # Test 6: Invalid URL
    total_tests += 1
    if test_invalid_url():
        tests_passed += 1
    
    # Test 7: List Jobs
    total_tests += 1
    if test_list_jobs_endpoint():
        tests_passed += 1
    
    # Results
    print("=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Phase 1 implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
