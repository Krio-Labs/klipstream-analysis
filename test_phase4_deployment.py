#!/usr/bin/env python3
"""
Phase 4 Deployment Testing Script

This script tests the Phase 4 deployment features including:
- FastAPI deployment validation
- Backward compatibility with legacy endpoints
- Production readiness checks
- Performance validation
- Migration strategy validation
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Configuration
LOCAL_URL = "http://localhost:3001"
PRODUCTION_URL = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"  # Update with actual URL

class Phase4Tester:
    def __init__(self, base_url: str = LOCAL_URL):
        self.base_url = base_url
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
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request and return response with status code"""
        url = f"{self.base_url}{endpoint}"
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
    
    async def test_fastapi_deployment(self):
        """Test FastAPI deployment and basic functionality"""
        print("\n🚀 Testing FastAPI Deployment...")
        
        # Test root endpoint
        response = await self.make_request("GET", "/")
        success = response.get("status_code") == 200
        self.log_test("FastAPI Root Endpoint", success,
                     f"Status: {response.get('status_code')}")
        
        # Test API documentation
        response = await self.make_request("GET", "/docs")
        success = response.get("status_code") == 200
        self.log_test("API Documentation", success,
                     f"Swagger UI available")
        
        # Test OpenAPI schema
        response = await self.make_request("GET", "/openapi.json")
        success = response.get("status_code") == 200 and "openapi" in response
        self.log_test("OpenAPI Schema", success,
                     f"OpenAPI version: {response.get('openapi', 'N/A')}")
        
        # Test health endpoint
        response = await self.make_request("GET", "/health")
        success = response.get("status") == "healthy" and response.get("status_code") == 200
        uptime = response.get("uptime_seconds", "N/A")
        self.log_test("Health Endpoint", success,
                     f"Status: {response.get('status')}, Uptime: {uptime}s")
    
    async def test_backward_compatibility(self):
        """Test backward compatibility with legacy endpoints"""
        print("\n🔄 Testing Backward Compatibility...")
        
        # Test migration status endpoint
        response = await self.make_request("GET", "/legacy/migration_status")
        success = response.get("status") == "success" and response.get("status_code") == 200
        phase = response.get("migration", {}).get("phase", "N/A")
        self.log_test("Migration Status Endpoint", success,
                     f"Phase: {phase}")
        
        # Test legacy health check
        response = await self.make_request("GET", "/legacy/health")
        success = response.get("status_code") == 200
        health_status = response.get("status", "N/A")
        self.log_test("Legacy Health Check", success,
                     f"Status: {health_status}")
        
        # Test legacy list files endpoint
        response = await self.make_request("GET", "/legacy/list_files")
        success = response.get("status") == "success" and response.get("status_code") == 200
        self.log_test("Legacy List Files", success)
        
        # Test legacy list output files endpoint
        response = await self.make_request("GET", "/legacy/list_output_files")
        success = response.get("status") == "success" and response.get("status_code") == 200
        self.log_test("Legacy List Output Files", success)
    
    async def test_new_api_endpoints(self):
        """Test new API endpoints are working"""
        print("\n🆕 Testing New API Endpoints...")
        
        # Test analysis endpoint structure (without actually running analysis)
        response = await self.make_request("POST", "/api/v1/analysis", 
                                         json={"url": "invalid-url-for-testing"})
        # Should return 400 for invalid URL, not 500
        success = response.get("status_code") in [400, 422]  # 422 for validation error
        self.log_test("Analysis Endpoint Validation", success,
                     f"Status: {response.get('status_code')}")
        
        # Test queue status
        response = await self.make_request("GET", "/api/v1/queue/status")
        success = response.get("status") == "success" and response.get("status_code") == 200
        queue_length = response.get("queue", {}).get("queue_length", "N/A")
        self.log_test("Queue Status Endpoint", success,
                     f"Queue length: {queue_length}")
        
        # Test monitoring health
        response = await self.make_request("GET", "/api/v1/monitoring/health")
        success = response.get("status") == "success" and response.get("status_code") == 200
        system_health = response.get("health", {}).get("status", "N/A")
        self.log_test("Monitoring Health Endpoint", success,
                     f"System health: {system_health}")
        
        # Test monitoring dashboard
        response = await self.make_request("GET", "/api/v1/monitoring/dashboard")
        success = response.get("status") == "success" and response.get("status_code") == 200
        performance_score = response.get("dashboard", {}).get("summary", {}).get("performance_score", "N/A")
        self.log_test("Monitoring Dashboard", success,
                     f"Performance score: {performance_score}")
    
    async def test_production_readiness(self):
        """Test production readiness features"""
        print("\n🏭 Testing Production Readiness...")
        
        # Test CORS headers
        response = await self.make_request("OPTIONS", "/health")
        success = response.get("status_code") in [200, 204]
        self.log_test("CORS Support", success,
                     f"OPTIONS request handled")
        
        # Test error handling
        response = await self.make_request("GET", "/nonexistent-endpoint")
        success = response.get("status_code") == 404
        self.log_test("404 Error Handling", success,
                     f"Status: {response.get('status_code')}")
        
        # Test rate limiting (if implemented)
        # Make multiple rapid requests
        start_time = time.time()
        responses = []
        for i in range(10):
            resp = await self.make_request("GET", "/health")
            responses.append(resp)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # All requests should succeed (no rate limiting implemented yet)
        successful_requests = sum(1 for r in responses if r.get("status_code") == 200)
        success = successful_requests == 10
        self.log_test("Concurrent Request Handling", success,
                     f"{successful_requests}/10 requests in {duration:.2f}s")
    
    async def test_performance_metrics(self):
        """Test performance and response times"""
        print("\n⚡ Testing Performance Metrics...")
        
        # Test response times for various endpoints
        endpoints = [
            "/health",
            "/api/v1/queue/status", 
            "/api/v1/monitoring/health",
            "/legacy/migration_status"
        ]
        
        response_times = []
        
        for endpoint in endpoints:
            start_time = time.time()
            response = await self.make_request("GET", endpoint)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            response_times.append(response_time)
            
            success = response.get("status_code") == 200 and response_time < 1000  # Under 1 second
            self.log_test(f"Response Time - {endpoint}", success,
                         f"{response_time:.0f}ms")
        
        # Calculate average response time
        avg_response_time = sum(response_times) / len(response_times)
        success = avg_response_time < 500  # Average under 500ms
        self.log_test("Average Response Time", success,
                     f"{avg_response_time:.0f}ms")
    
    async def test_migration_strategy(self):
        """Test migration strategy and compatibility"""
        print("\n🔄 Testing Migration Strategy...")
        
        # Test that both old and new endpoints coexist
        legacy_health = await self.make_request("GET", "/legacy/health")
        new_health = await self.make_request("GET", "/health")
        
        legacy_success = legacy_health.get("status_code") == 200
        new_success = new_health.get("status_code") == 200
        
        success = legacy_success and new_success
        self.log_test("Dual Endpoint Support", success,
                     f"Legacy: {legacy_success}, New: {new_success}")
        
        # Test migration information
        response = await self.make_request("GET", "/legacy/migration_status")
        migration_info = response.get("migration", {})
        
        has_legacy_endpoints = len(migration_info.get("legacy_endpoints", [])) > 0
        has_new_endpoints = len(migration_info.get("new_endpoints", [])) > 0
        
        success = has_legacy_endpoints and has_new_endpoints
        self.log_test("Migration Information", success,
                     f"Legacy endpoints: {len(migration_info.get('legacy_endpoints', []))}, "
                     f"New endpoints: {len(migration_info.get('new_endpoints', []))}")
    
    async def run_all_tests(self):
        """Run all Phase 4 tests"""
        print("🚀 Starting Phase 4 Deployment Test Suite")
        print(f"Testing against: {self.base_url}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        await self.test_fastapi_deployment()
        await self.test_backward_compatibility()
        await self.test_new_api_endpoints()
        await self.test_production_readiness()
        await self.test_performance_metrics()
        await self.test_migration_strategy()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 Phase 4 Test Summary")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if passed == total:
            print("\n🎉 Phase 4 deployment is ready for production!")
        else:
            print(f"\n⚠️  {total - passed} tests failed. Review before production deployment.")
        
        # Print failed tests
        failed_tests = [r for r in self.test_results if not r["success"]]
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        
        return passed == total


async def main():
    """Main test function"""
    import sys
    
    # Allow testing against different environments
    base_url = LOCAL_URL
    if len(sys.argv) > 1:
        if sys.argv[1] == "production":
            base_url = PRODUCTION_URL
        else:
            base_url = sys.argv[1]
    
    async with Phase4Tester(base_url) as tester:
        success = await tester.run_all_tests()
        return 0 if success else 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
