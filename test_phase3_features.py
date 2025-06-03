#!/usr/bin/env python3
"""
Test Script for Phase 3 Features

This script tests the enhanced features implemented in Phase 3:
- Webhook system with registration and delivery
- Enhanced health checks and monitoring
- Performance metrics and caching
- Production-ready features
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
TEST_WEBHOOK_URL = "https://httpbin.org/post"  # Test webhook endpoint


def test_webhook_registration():
    """Test webhook registration and management"""
    print("🔍 Testing Webhook Registration...")
    
    try:
        # Test webhook registration
        webhook_data = {
            "url": TEST_WEBHOOK_URL,
            "events": ["job.started", "job.completed", "job.failed"],
            "secret": "test-secret-key",
            "timeout_seconds": 30,
            "max_retries": 3
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/webhooks?webhook_id=test-webhook-1",
            json=webhook_data
        )
        assert response.status_code == 201
        
        webhook_info = response.json()
        print(f"✅ Webhook registered: {webhook_info['id']}")
        print(f"   URL: {webhook_info['url']}")
        print(f"   Events: {webhook_info['events']}")
        
        # Test webhook listing
        list_response = requests.get(f"{API_BASE_URL}/api/v1/webhooks")
        assert list_response.status_code == 200
        
        webhooks = list_response.json()
        print(f"✅ Found {len(webhooks)} registered webhooks")
        
        # Test webhook update
        update_data = {
            "events": ["job.started", "job.completed", "job.failed", "job.progress"],
            "is_active": True
        }
        
        update_response = requests.put(
            f"{API_BASE_URL}/api/v1/webhooks/test-webhook-1",
            json=update_data
        )
        assert update_response.status_code == 200
        
        updated_webhook = update_response.json()
        print(f"✅ Webhook updated: {len(updated_webhook['events'])} events")
        
        return True
        
    except Exception as e:
        print(f"❌ Webhook registration test failed: {str(e)}")
        return False


def test_webhook_delivery():
    """Test webhook delivery with a real job"""
    print("🔍 Testing Webhook Delivery...")
    
    try:
        # Start an analysis job to trigger webhooks
        payload = {"url": TEST_VIDEO_URL}
        response = requests.post(f"{API_BASE_URL}/api/v1/analysis", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        job_id = data["job_id"]
        print(f"✅ Started analysis job: {job_id}")
        
        # Wait a bit for webhook delivery
        time.sleep(10)
        
        # Check webhook delivery logs
        logs_response = requests.get(f"{API_BASE_URL}/api/v1/webhooks/test-webhook-1/logs")
        assert logs_response.status_code == 200
        
        logs = logs_response.json()
        print(f"✅ Found {len(logs)} webhook delivery attempts")
        
        if logs:
            latest_log = logs[0]
            print(f"   Latest delivery: {latest_log['event']} - {latest_log['status']}")
            print(f"   Attempt count: {latest_log['attempt_count']}")
        
        # Check webhook statistics
        stats_response = requests.get(f"{API_BASE_URL}/api/v1/webhooks/test-webhook-1/stats")
        assert stats_response.status_code == 200
        
        stats = stats_response.json()
        print(f"✅ Webhook stats:")
        print(f"   Total deliveries: {stats['total_deliveries']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Webhook delivery test failed: {str(e)}")
        return False


def test_webhook_testing():
    """Test webhook testing functionality"""
    print("🔍 Testing Webhook Test Feature...")
    
    try:
        # Test webhook delivery
        test_data = {
            "event": "job.started",
            "test_payload": {
                "message": "This is a test webhook",
                "test_timestamp": datetime.utcnow().isoformat()
            }
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/webhooks/test-webhook-1/test",
            json=test_data
        )
        assert response.status_code == 202
        
        result = response.json()
        print(f"✅ Test webhook sent: {result['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Webhook testing failed: {str(e)}")
        return False


def test_enhanced_health_checks():
    """Test enhanced health check endpoints"""
    print("🔍 Testing Enhanced Health Checks...")
    
    try:
        # Test basic health check
        health_response = requests.get(f"{API_BASE_URL}/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        print(f"✅ Basic health check: {health_data['status']}")
        print(f"   Uptime: {health_data['uptime_seconds']:.1f} seconds")
        
        # Test detailed health check
        detailed_response = requests.get(f"{API_BASE_URL}/health/detailed")
        assert detailed_response.status_code == 200
        
        detailed_data = detailed_response.json()
        print(f"✅ Detailed health check: {detailed_data['status']}")
        
        # Check system information
        system_info = detailed_data['system']
        print(f"   Memory usage: {system_info['memory']['percent_used']:.1f}%")
        print(f"   CPU usage: {system_info['cpu']['percent_used']:.1f}%")
        
        # Check services
        services = detailed_data['services']
        for service_name, service_info in services.items():
            status = service_info.get('status', 'unknown')
            print(f"   {service_name}: {status}")
        
        # Test readiness check
        readiness_response = requests.get(f"{API_BASE_URL}/health/readiness")
        assert readiness_response.status_code == 200
        
        readiness_data = readiness_response.json()
        print(f"✅ Readiness check: {readiness_data['status']}")
        
        # Test liveness check
        liveness_response = requests.get(f"{API_BASE_URL}/health/liveness")
        assert liveness_response.status_code == 200
        
        liveness_data = liveness_response.json()
        print(f"✅ Liveness check: {liveness_data['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced health checks test failed: {str(e)}")
        return False


def test_metrics_endpoint():
    """Test metrics collection and endpoint"""
    print("🔍 Testing Metrics Endpoint...")
    
    try:
        # Test metrics endpoint
        metrics_response = requests.get(f"{API_BASE_URL}/metrics")
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.json()
        print(f"✅ Metrics endpoint accessible")
        print(f"   Format: {metrics_data['format']}")
        
        # Check if metrics contain expected data
        metrics_content = metrics_data['metrics']
        if "jobs_created_total" in metrics_content:
            print("✅ Found business metrics")
        
        if "memory_usage_bytes" in metrics_content:
            print("✅ Found system metrics")
        
        if "api_request_duration_seconds" in metrics_content:
            print("✅ Found performance metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics endpoint test failed: {str(e)}")
        return False


def test_available_webhook_events():
    """Test available webhook events endpoint"""
    print("🔍 Testing Available Webhook Events...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/webhooks/events/available")
        assert response.status_code == 200
        
        events = response.json()
        print(f"✅ Found {len(events)} available webhook events:")
        
        for event in events:
            print(f"   {event['value']}: {event['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Available webhook events test failed: {str(e)}")
        return False


def test_webhook_error_handling():
    """Test webhook error handling with invalid data"""
    print("🔍 Testing Webhook Error Handling...")
    
    try:
        # Test invalid webhook URL
        invalid_webhook_data = {
            "url": "not-a-valid-url",
            "events": ["job.started"]
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/webhooks?webhook_id=invalid-webhook",
            json=invalid_webhook_data
        )
        assert response.status_code == 400
        print("✅ Invalid URL properly rejected")
        
        # Test invalid events
        invalid_events_data = {
            "url": "https://example.com/webhook",
            "events": ["invalid.event", "job.started"]
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/webhooks?webhook_id=invalid-events",
            json=invalid_events_data
        )
        assert response.status_code == 422  # Validation error
        print("✅ Invalid events properly rejected")
        
        # Test duplicate webhook ID
        valid_webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["job.started"]
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/webhooks?webhook_id=test-webhook-1",
            json=valid_webhook_data
        )
        assert response.status_code == 409  # Conflict
        print("✅ Duplicate webhook ID properly rejected")
        
        return True
        
    except Exception as e:
        print(f"❌ Webhook error handling test failed: {str(e)}")
        return False


def test_webhook_cleanup():
    """Clean up test webhooks"""
    print("🔍 Cleaning up test webhooks...")
    
    try:
        # Delete test webhook
        response = requests.delete(f"{API_BASE_URL}/api/v1/webhooks/test-webhook-1")
        assert response.status_code == 204
        print("✅ Test webhook deleted")
        
        # Verify deletion
        get_response = requests.get(f"{API_BASE_URL}/api/v1/webhooks/test-webhook-1")
        assert get_response.status_code == 404
        print("✅ Webhook deletion verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Webhook cleanup failed: {str(e)}")
        return False


def main():
    """Run all Phase 3 tests"""
    print("🚀 Starting Phase 3 Feature Tests")
    print("=" * 50)
    
    # Check if API is running
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=5)
    except requests.exceptions.RequestException:
        print("❌ API is not running!")
        print("   Start the API with: python -m api.main")
        return False
    
    # Run tests in order
    tests = [
        ("Available Webhook Events", test_available_webhook_events),
        ("Webhook Registration", test_webhook_registration),
        ("Webhook Error Handling", test_webhook_error_handling),
        ("Webhook Testing", test_webhook_testing),
        ("Webhook Delivery", test_webhook_delivery),
        ("Enhanced Health Checks", test_enhanced_health_checks),
        ("Metrics Endpoint", test_metrics_endpoint),
        ("Webhook Cleanup", test_webhook_cleanup),
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
    print("🎯 Phase 3 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 3 tests passed! Production-ready features are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
