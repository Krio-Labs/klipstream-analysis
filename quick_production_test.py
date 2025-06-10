#!/usr/bin/env python3
"""
Quick Production Test Script for KlipStream Analysis API

This script runs essential production readiness tests quickly to validate:
- Service health and response times
- Basic functionality
- Error handling
- Performance baseline
"""

import asyncio
import aiohttp
import time
import json
import sys
from datetime import datetime
from typing import Dict, Any

class QuickProductionTest:
    """Quick production test suite for immediate validation"""
    
    def __init__(self, base_url: str = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"):
        self.base_url = base_url
        self.results = []
    
    def log_result(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
        
        self.results.append({
            "test": test_name,
            "success": success,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
    
    async def test_health_check(self) -> bool:
        """Test basic health endpoint"""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    response_time = (time.time() - start_time) * 1000
                    data = await response.json()
                    
                    success = response.status == 200 and response_time < 5000
                    self.log_result("Health Check", success, {
                        "status_code": response.status,
                        "response_time_ms": f"{response_time:.1f}",
                        "service_status": data.get("status"),
                        "version": data.get("version")
                    })
                    return success
        except Exception as e:
            self.log_result("Health Check", False, {"error": str(e)})
            return False
    
    async def test_api_discovery(self) -> bool:
        """Test API discovery endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    data = await response.json()
                    
                    success = response.status == 200 and "endpoints" in data
                    self.log_result("API Discovery", success, {
                        "status_code": response.status,
                        "api_name": data.get("name"),
                        "version": data.get("version"),
                        "endpoints_available": len(data.get("endpoints", {}))
                    })
                    return success
        except Exception as e:
            self.log_result("API Discovery", False, {"error": str(e)})
            return False
    
    async def test_analysis_endpoint(self) -> bool:
        """Test analysis endpoint validation"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test with invalid data to check validation
                async with session.post(
                    f"{self.base_url}/api/v1/analysis",
                    json={"invalid": "data"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    success = response.status in [400, 422]  # Validation error expected
                    self.log_result("Analysis Endpoint Validation", success, {
                        "status_code": response.status,
                        "validation_working": success
                    })
                    return success
        except Exception as e:
            self.log_result("Analysis Endpoint Validation", False, {"error": str(e)})
            return False
    
    async def test_concurrent_requests(self) -> bool:
        """Test concurrent request handling"""
        try:
            async def make_request(session):
                start_time = time.time()
                async with session.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return {
                        "success": response.status == 200,
                        "response_time": (time.time() - start_time) * 1000
                    }
            
            async with aiohttp.ClientSession() as session:
                # Make 5 concurrent requests
                tasks = [make_request(session) for _ in range(5)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
                avg_time = sum(r.get("response_time", 0) for r in results if isinstance(r, dict)) / len(results)
                
                success = successful >= 4  # 80% success rate
                self.log_result("Concurrent Requests", success, {
                    "successful_requests": f"{successful}/5",
                    "success_rate": f"{successful/5:.1%}",
                    "avg_response_time_ms": f"{avg_time:.1f}"
                })
                return success
        except Exception as e:
            self.log_result("Concurrent Requests", False, {"error": str(e)})
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/non-existent", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    success = response.status == 404
                    self.log_result("Error Handling", success, {
                        "status_code": response.status,
                        "proper_404": success
                    })
                    return success
        except Exception as e:
            self.log_result("Error Handling", False, {"error": str(e)})
            return False
    
    async def test_detailed_health(self) -> bool:
        """Test detailed health endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health/detailed", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        success = "system" in data and "services" in data
                        self.log_result("Detailed Health", success, {
                            "status_code": response.status,
                            "has_system_info": "system" in data,
                            "has_services_info": "services" in data
                        })
                        return success
                    else:
                        self.log_result("Detailed Health", False, {"status_code": response.status})
                        return False
        except Exception as e:
            self.log_result("Detailed Health", False, {"error": str(e)})
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all quick tests"""
        print("🚀 Running Quick Production Test Suite")
        print(f"🌐 Testing: {self.base_url}")
        print("=" * 60)
        
        tests = [
            self.test_health_check,
            self.test_api_discovery,
            self.test_analysis_endpoint,
            self.test_concurrent_requests,
            self.test_error_handling,
            self.test_detailed_health
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if await test():
                    passed += 1
                await asyncio.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                print(f"❌ Test failed with exception: {str(e)}")
        
        success_rate = passed / total
        overall_status = "READY" if success_rate >= 0.8 else "NOT_READY"
        
        print("\n" + "=" * 60)
        print(f"🎯 QUICK TEST RESULTS: {overall_status}")
        print(f"📊 Passed: {passed}/{total} ({success_rate:.1%})")
        
        if overall_status == "READY":
            print("✅ Service appears ready for production traffic!")
        else:
            print("❌ Service needs attention before production deployment")
        
        return {
            "overall_status": overall_status,
            "passed": passed,
            "total": total,
            "success_rate": success_rate,
            "results": self.results,
            "timestamp": datetime.now().isoformat(),
            "service_url": self.base_url
        }


async def main():
    """Main execution"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"
    
    test_suite = QuickProductionTest(base_url)
    results = await test_suite.run_all_tests()
    
    # Save results
    with open("quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: quick_test_results.json")
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] == "READY" else 1)


if __name__ == "__main__":
    asyncio.run(main())
