#!/usr/bin/env python3
"""
Test Script for Phase 2 Features

This script tests the enhanced features implemented in Phase 2:
- Enhanced Server-Sent Events
- Error classification and handling
- Retry mechanisms
- Pipeline integration improvements
"""

import asyncio
import json
import time
import requests
import threading
from datetime import datetime
from typing import List, Dict

# Test configuration
API_BASE_URL = "http://localhost:3000"
TEST_VIDEO_URL = "https://www.twitch.tv/videos/2434635255"
INVALID_VIDEO_URL = "https://www.twitch.tv/videos/999999999"


class SSEClient:
    """Simple SSE client for testing"""
    
    def __init__(self, url: str):
        self.url = url
        self.events: List[Dict] = []
        self.is_running = False
        self.thread = None
    
    def start(self):
        """Start listening to SSE stream"""
        self.is_running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.start()
    
    def stop(self):
        """Stop listening to SSE stream"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _listen(self):
        """Listen to SSE stream"""
        try:
            response = requests.get(self.url, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if not self.is_running:
                    break
                
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        self.events.append({
                            'timestamp': datetime.now().isoformat(),
                            'data': data
                        })
                        print(f"📡 SSE Event: {data.get('event', 'unknown')} - {data.get('message', 'no message')}")
                    except json.JSONDecodeError:
                        print(f"📡 SSE Raw: {line}")
                
        except Exception as e:
            print(f"❌ SSE Error: {str(e)}")


def test_enhanced_sse():
    """Test enhanced Server-Sent Events functionality"""
    print("🔍 Testing Enhanced Server-Sent Events...")
    
    try:
        # Start an analysis job
        payload = {"url": TEST_VIDEO_URL}
        response = requests.post(f"{API_BASE_URL}/api/v1/analysis", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        job_id = data["job_id"]
        print(f"✅ Started analysis job: {job_id}")
        
        # Connect to SSE stream
        sse_url = f"{API_BASE_URL}/api/v1/analysis/{job_id}/stream"
        sse_client = SSEClient(sse_url)
        
        print(f"📡 Connecting to SSE stream: {sse_url}")
        sse_client.start()
        
        # Listen for 30 seconds
        time.sleep(30)
        
        # Stop SSE client
        sse_client.stop()
        
        # Analyze events
        print(f"📊 Received {len(sse_client.events)} SSE events")
        
        if sse_client.events:
            print("📋 Event Summary:")
            for i, event in enumerate(sse_client.events[:10]):  # Show first 10 events
                event_type = event['data'].get('event', 'unknown')
                progress = event['data'].get('progress', {}).get('percentage', 0)
                print(f"  {i+1}. {event_type} - {progress:.1f}%")
            
            # Check for expected events
            event_types = [event['data'].get('event') for event in sse_client.events]
            expected_events = ['connected', 'progress']
            
            for expected in expected_events:
                if expected in event_types:
                    print(f"✅ Found expected event: {expected}")
                else:
                    print(f"⚠️  Missing expected event: {expected}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced SSE test failed: {str(e)}")
        return False


def test_error_classification():
    """Test error classification with invalid URLs"""
    print("🔍 Testing Error Classification...")
    
    try:
        # Test with invalid video URL
        payload = {"url": INVALID_VIDEO_URL}
        response = requests.post(f"{API_BASE_URL}/api/v1/analysis", json=payload)
        
        # Should still return 200 (job created) but will fail during processing
        assert response.status_code == 200
        
        data = response.json()
        job_id = data["job_id"]
        print(f"✅ Started job with invalid URL: {job_id}")
        
        # Wait for processing to fail
        time.sleep(10)
        
        # Check job status
        status_response = requests.get(f"{API_BASE_URL}/api/v1/analysis/{job_id}/status")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        print(f"📊 Job status: {status_data['status']}")
        
        if status_data['status'] == 'Failed' and status_data.get('error'):
            error = status_data['error']
            print(f"✅ Error classified as: {error.get('error_type')}")
            print(f"📝 Error message: {error.get('error_message')}")
            print(f"🔄 Is retryable: {error.get('is_retryable')}")
            print(f"💡 Suggested action: {error.get('suggested_action')}")
            return True
        else:
            print("⚠️  Job hasn't failed yet or no error info available")
            return False
        
    except Exception as e:
        print(f"❌ Error classification test failed: {str(e)}")
        return False


def test_malformed_url():
    """Test with completely malformed URL"""
    print("🔍 Testing Malformed URL Handling...")
    
    try:
        payload = {"url": "not-a-valid-url"}
        response = requests.post(f"{API_BASE_URL}/api/v1/analysis", json=payload)
        
        # Should return 400 for malformed URL
        assert response.status_code == 400
        
        data = response.json()
        print(f"✅ Malformed URL properly rejected")
        print(f"📝 Error message: {data['detail']['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Malformed URL test failed: {str(e)}")
        return False


def test_job_status_details():
    """Test detailed job status information"""
    print("🔍 Testing Detailed Job Status...")
    
    try:
        # Start a job
        payload = {"url": TEST_VIDEO_URL}
        response = requests.post(f"{API_BASE_URL}/api/v1/analysis", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        job_id = data["job_id"]
        
        # Check status multiple times
        for i in range(5):
            time.sleep(2)
            
            status_response = requests.get(f"{API_BASE_URL}/api/v1/analysis/{job_id}/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            progress = status_data.get('progress', {})
            
            print(f"📊 Check {i+1}: {status_data['status']} - {progress.get('percentage', 0):.1f}%")
            print(f"   Stage: {progress.get('current_stage', 'unknown')}")
            print(f"   ETA: {progress.get('estimated_completion_seconds', 0)} seconds")
            
            if status_data['status'] in ['Completed', 'Failed']:
                break
        
        return True
        
    except Exception as e:
        print(f"❌ Job status details test failed: {str(e)}")
        return False


def test_concurrent_jobs():
    """Test multiple concurrent jobs"""
    print("🔍 Testing Concurrent Jobs...")
    
    try:
        job_ids = []
        
        # Start multiple jobs
        for i in range(3):
            payload = {"url": TEST_VIDEO_URL}
            response = requests.post(f"{API_BASE_URL}/api/v1/analysis", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            job_ids.append(data["job_id"])
            print(f"✅ Started job {i+1}: {data['job_id']}")
        
        # Check all jobs are listed
        jobs_response = requests.get(f"{API_BASE_URL}/api/v1/jobs")
        assert jobs_response.status_code == 200
        
        jobs_data = jobs_response.json()
        active_jobs = [job['job_id'] for job in jobs_data['jobs']]
        
        print(f"📊 Found {len(active_jobs)} active jobs")
        
        # Verify our jobs are in the list
        for job_id in job_ids:
            if job_id in active_jobs:
                print(f"✅ Job {job_id} found in active jobs")
            else:
                print(f"⚠️  Job {job_id} not found in active jobs")
        
        return True
        
    except Exception as e:
        print(f"❌ Concurrent jobs test failed: {str(e)}")
        return False


def test_api_documentation():
    """Test API documentation endpoints"""
    print("🔍 Testing API Documentation...")
    
    try:
        # Test OpenAPI docs
        docs_response = requests.get(f"{API_BASE_URL}/docs")
        assert docs_response.status_code == 200
        print("✅ OpenAPI docs accessible")
        
        # Test ReDoc
        redoc_response = requests.get(f"{API_BASE_URL}/redoc")
        assert redoc_response.status_code == 200
        print("✅ ReDoc documentation accessible")
        
        # Test OpenAPI JSON
        openapi_response = requests.get(f"{API_BASE_URL}/openapi.json")
        assert openapi_response.status_code == 200
        
        openapi_data = openapi_response.json()
        print(f"✅ OpenAPI spec loaded - {len(openapi_data.get('paths', {}))} endpoints documented")
        
        return True
        
    except Exception as e:
        print(f"❌ API documentation test failed: {str(e)}")
        return False


def main():
    """Run all Phase 2 tests"""
    print("🚀 Starting Phase 2 Feature Tests")
    print("=" * 50)
    
    # Check if API is running
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=5)
    except requests.exceptions.RequestException:
        print("❌ API is not running!")
        print("   Start the API with: python -m api.main")
        return False
    
    # Run tests
    tests = [
        ("API Documentation", test_api_documentation),
        ("Malformed URL Handling", test_malformed_url),
        ("Error Classification", test_error_classification),
        ("Detailed Job Status", test_job_status_details),
        ("Concurrent Jobs", test_concurrent_jobs),
        ("Enhanced Server-Sent Events", test_enhanced_sse),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"⚠️  {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Phase 2 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 tests passed! Enhanced features are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
