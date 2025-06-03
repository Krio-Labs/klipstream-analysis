#!/usr/bin/env python3
"""
Comprehensive Test Script for Phase 3 Features

This script tests all the new Phase 3 features including:
- Queue Management System
- Enhanced Monitoring and Analytics
- Performance Metrics
- Alert System
- Dashboard APIs
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:3001"
API_BASE = f"{BASE_URL}/api/v1"

class Phase3Tester:
    def __init__(self):
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": timestamp
        })
    
    async def make_request(self, method: str, endpoint: str, use_api_base: bool = True, **kwargs) -> Dict[str, Any]:
        """Make HTTP request and return JSON response with status code"""
        if use_api_base:
            url = f"{API_BASE}{endpoint}"
        else:
            url = f"{BASE_URL}{endpoint}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                result = {"status_code": response.status}

                if response.content_type == 'application/json':
                    json_data = await response.json()
                    result.update(json_data)
                else:
                    result["text"] = await response.text()

                return result
        except Exception as e:
            return {"error": str(e), "status_code": 500}
    
    async def test_queue_management(self):
        """Test Queue Management System"""
        print("\n🔄 Testing Queue Management System...")
        
        # Test queue status
        response = await self.make_request("GET", "/queue/status")
        success = response.get("status") == "success" and "queue" in response
        self.log_test("Queue Status API", success, 
                     f"Queue length: {response.get('queue', {}).get('queue_length', 'N/A')}")
        
        # Test queue metrics
        response = await self.make_request("GET", "/queue/metrics")
        success = response.get("status") == "success" and "metrics" in response
        metrics = response.get("metrics", {})
        self.log_test("Queue Metrics API", success,
                     f"Utilization: {metrics.get('utilization_percent', 'N/A')}%")
        
        # Test queue health
        response = await self.make_request("GET", "/queue/health")
        success = response.get("status") == "success" and "health" in response
        health = response.get("health", {})
        self.log_test("Queue Health API", success,
                     f"Status: {health.get('status', 'N/A')}")
        
        # Test queue pause/resume
        response = await self.make_request("POST", "/queue/pause")
        success = response.get("status") == "success"
        self.log_test("Queue Pause API", success)
        
        response = await self.make_request("POST", "/queue/resume")
        success = response.get("status") == "success"
        self.log_test("Queue Resume API", success)
    
    async def test_monitoring_system(self):
        """Test Enhanced Monitoring System"""
        print("\n📊 Testing Enhanced Monitoring System...")
        
        # Test system metrics
        response = await self.make_request("GET", "/monitoring/metrics")
        success = response.get("status") == "success" and "metrics" in response
        metrics = response.get("metrics", {})
        cpu = metrics.get("cpu_percent", {}).get("current", "N/A")
        memory = metrics.get("memory_percent", {}).get("current", "N/A")
        self.log_test("System Metrics API", success,
                     f"CPU: {cpu}%, Memory: {memory}%")
        
        # Test system health
        response = await self.make_request("GET", "/monitoring/health")
        success = response.get("status") == "success" and "health" in response
        health_status = response.get("health", {}).get("status", "N/A")
        self.log_test("System Health API", success,
                     f"Health: {health_status}")
        
        # Test alerts
        response = await self.make_request("GET", "/monitoring/alerts")
        success = response.get("status") == "success" and "alerts" in response
        alert_count = len(response.get("alerts", []))
        self.log_test("Alerts API", success,
                     f"Active alerts: {alert_count}")
        
        # Test custom metrics
        custom_metric = {
            "name": "test_metric",
            "value": 42.5,
            "labels": {"test": "phase3"}
        }
        response = await self.make_request("POST", "/monitoring/metrics/custom", 
                                         json=custom_metric)
        success = response.get("status") == "success"
        self.log_test("Custom Metrics API", success)
        
        # Test performance analytics
        response = await self.make_request("GET", "/monitoring/performance")
        success = response.get("status") == "success" and "performance" in response
        performance = response.get("performance", {})
        score = performance.get("score", "N/A")
        grade = performance.get("grade", "N/A")
        self.log_test("Performance Analytics API", success,
                     f"Score: {score}, Grade: {grade}")
    
    async def test_dashboard_apis(self):
        """Test Dashboard APIs"""
        print("\n📈 Testing Dashboard APIs...")
        
        # Test monitoring dashboard
        response = await self.make_request("GET", "/monitoring/dashboard")
        success = response.get("status") == "success" and "dashboard" in response
        dashboard = response.get("dashboard", {})
        system_status = dashboard.get("health", {}).get("status", "N/A")
        performance_score = dashboard.get("summary", {}).get("performance_score", "N/A")
        self.log_test("Monitoring Dashboard API", success,
                     f"System: {system_status}, Performance: {performance_score}")
        
        # Test comprehensive health check
        response = await self.make_request("GET", "/health/detailed", use_api_base=False)
        success = response.get("status") == "healthy" and response.get("status_code") == 200
        health_status = response.get("status", "N/A")
        uptime = response.get("uptime_seconds", "N/A")
        self.log_test("Detailed Health Check API", success,
                     f"Health: {health_status}, Uptime: {uptime}s")
    
    async def test_api_integration(self):
        """Test API Integration and Error Handling"""
        print("\n🔗 Testing API Integration...")
        
        # Test invalid endpoints
        response = await self.make_request("GET", "/queue/invalid")
        success = response.get("status_code") == 404
        self.log_test("404 Error Handling", success,
                     f"Status: {response.get('status_code')}, Detail: {response.get('detail', 'N/A')}")

        # Test invalid queue job removal
        response = await self.make_request("DELETE", "/queue/jobs/invalid-job-id")
        success = response.get("status_code") == 404
        self.log_test("Invalid Job Removal Handling", success,
                     f"Status: {response.get('status_code')}, Detail: {response.get('detail', 'N/A')}")

        # Test invalid custom metric
        invalid_metric = {"name": "test", "invalid": "data"}
        response = await self.make_request("POST", "/monitoring/metrics/custom",
                                         json=invalid_metric)
        success = response.get("status_code") == 400
        self.log_test("Invalid Metric Handling", success,
                     f"Status: {response.get('status_code')}, Detail: {response.get('detail', 'N/A')}")
    
    async def test_performance_load(self):
        """Test Performance Under Load"""
        print("\n⚡ Testing Performance Under Load...")
        
        start_time = time.time()
        
        # Make multiple concurrent requests
        tasks = []
        for i in range(10):
            tasks.append(self.make_request("GET", "/monitoring/metrics"))
            tasks.append(self.make_request("GET", "/queue/status"))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if all requests succeeded
        successful_requests = sum(1 for r in responses if isinstance(r, dict) and r.get("status") == "success")
        total_requests = len(tasks)
        
        success = successful_requests == total_requests and duration < 5.0  # Should complete in under 5 seconds
        self.log_test("Concurrent Request Handling", success,
                     f"{successful_requests}/{total_requests} requests in {duration:.2f}s")
    
    async def run_all_tests(self):
        """Run all Phase 3 tests"""
        print("🚀 Starting Phase 3 Comprehensive Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        await self.test_queue_management()
        await self.test_monitoring_system()
        await self.test_dashboard_apis()
        await self.test_api_integration()
        await self.test_performance_load()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 Test Summary")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if passed == total:
            print("\n🎉 All Phase 3 features are working perfectly!")
        else:
            print(f"\n⚠️  {total - passed} tests failed. Check the details above.")
        
        # Print failed tests
        failed_tests = [r for r in self.test_results if not r["success"]]
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        
        return passed == total


async def main():
    """Main test function"""
    async with Phase3Tester() as tester:
        success = await tester.run_all_tests()
        return 0 if success else 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
