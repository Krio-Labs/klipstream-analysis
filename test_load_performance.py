#!/usr/bin/env python3
"""
Load Testing Script for Phase 4 Deployment

This script performs load testing to validate the production readiness
of the FastAPI deployment under various load conditions.
"""

import asyncio
import aiohttp
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

# Configuration
BASE_URL = "http://localhost:3001"

@dataclass
class LoadTestResult:
    endpoint: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    requests_per_second: float
    error_rate: float


class LoadTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
    
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
        """Make a single request and measure response time"""
        start_time = time.time()
        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                end_time = time.time()
                return {
                    "success": True,
                    "status_code": response.status,
                    "response_time": end_time - start_time,
                    "error": None
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "status_code": 0,
                "response_time": end_time - start_time,
                "error": str(e)
            }
    
    async def load_test_endpoint(self, endpoint: str, concurrent_requests: int, total_requests: int) -> LoadTestResult:
        """Perform load test on a specific endpoint"""
        print(f"\n🔥 Load testing {endpoint}")
        print(f"   Concurrent requests: {concurrent_requests}")
        print(f"   Total requests: {total_requests}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def bounded_request(session):
            async with semaphore:
                return await self.make_request(session, endpoint)
        
        # Run the load test
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [bounded_request(session) for _ in range(total_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Process results
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        for result in results:
            if isinstance(result, dict):
                if result["success"]:
                    successful_requests += 1
                else:
                    failed_requests += 1
                response_times.append(result["response_time"])
            else:
                failed_requests += 1
                response_times.append(0)  # Failed request
        
        # Calculate statistics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Calculate 95th percentile
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p95_response_time = sorted_times[p95_index]
        else:
            p95_response_time = 0
        
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        # Create result object
        load_result = LoadTestResult(
            endpoint=endpoint,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate
        )
        
        self.results.append(load_result)
        
        # Print results
        print(f"   ✅ Completed in {total_duration:.2f}s")
        print(f"   📊 Success rate: {(successful_requests/total_requests)*100:.1f}%")
        print(f"   ⚡ Requests/sec: {requests_per_second:.1f}")
        print(f"   ⏱️  Avg response: {avg_response_time*1000:.0f}ms")
        print(f"   📈 P95 response: {p95_response_time*1000:.0f}ms")
        
        return load_result
    
    async def run_load_tests(self):
        """Run comprehensive load tests"""
        print("🚀 Starting Load Testing Suite")
        print("=" * 60)
        
        # Test scenarios
        test_scenarios = [
            # Light load
            {"endpoint": "/health", "concurrent": 5, "total": 50},
            {"endpoint": "/api/v1/queue/status", "concurrent": 5, "total": 50},
            
            # Medium load
            {"endpoint": "/health", "concurrent": 20, "total": 200},
            {"endpoint": "/api/v1/monitoring/health", "concurrent": 20, "total": 200},
            
            # Heavy load
            {"endpoint": "/health", "concurrent": 50, "total": 500},
            {"endpoint": "/api/v1/monitoring/dashboard", "concurrent": 30, "total": 300},
            
            # Stress test
            {"endpoint": "/health", "concurrent": 100, "total": 1000},
        ]
        
        # Run all test scenarios
        for scenario in test_scenarios:
            await self.load_test_endpoint(
                scenario["endpoint"],
                scenario["concurrent"],
                scenario["total"]
            )
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        # Print comprehensive summary
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 Load Testing Summary")
        print("=" * 60)
        
        if not self.results:
            print("No test results available")
            return
        
        # Overall statistics
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        overall_success_rate = (total_successful / total_requests) * 100 if total_requests > 0 else 0
        
        print(f"Total Requests: {total_requests}")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print()
        
        # Detailed results table
        print("Detailed Results:")
        print("-" * 120)
        print(f"{'Endpoint':<30} {'Requests':<10} {'Success%':<10} {'RPS':<10} {'Avg(ms)':<10} {'P95(ms)':<10} {'Max(ms)':<10}")
        print("-" * 120)
        
        for result in self.results:
            success_rate = (result.successful_requests / result.total_requests) * 100
            print(f"{result.endpoint:<30} {result.total_requests:<10} {success_rate:<10.1f} "
                  f"{result.requests_per_second:<10.1f} {result.avg_response_time*1000:<10.0f} "
                  f"{result.p95_response_time*1000:<10.0f} {result.max_response_time*1000:<10.0f}")
        
        print("-" * 120)
        
        # Performance assessment
        print("\n📈 Performance Assessment:")
        
        # Check if any tests failed
        failed_tests = [r for r in self.results if r.error_rate > 0.05]  # More than 5% error rate
        if failed_tests:
            print("❌ High error rate detected in:")
            for test in failed_tests:
                print(f"   - {test.endpoint}: {test.error_rate*100:.1f}% error rate")
        
        # Check response times
        slow_tests = [r for r in self.results if r.p95_response_time > 1.0]  # P95 > 1 second
        if slow_tests:
            print("⚠️  Slow response times detected in:")
            for test in slow_tests:
                print(f"   - {test.endpoint}: P95 = {test.p95_response_time*1000:.0f}ms")
        
        # Check throughput
        low_throughput = [r for r in self.results if r.requests_per_second < 10]  # Less than 10 RPS
        if low_throughput:
            print("📉 Low throughput detected in:")
            for test in low_throughput:
                print(f"   - {test.endpoint}: {test.requests_per_second:.1f} RPS")
        
        # Overall assessment
        if not failed_tests and not slow_tests:
            print("✅ All tests passed performance criteria!")
            print("🚀 System is ready for production load!")
        else:
            print("⚠️  Some performance issues detected. Review before production deployment.")
        
        # Recommendations
        print("\n💡 Recommendations:")
        
        avg_p95 = statistics.mean([r.p95_response_time for r in self.results])
        if avg_p95 > 0.5:  # Average P95 > 500ms
            print("   - Consider optimizing response times")
            print("   - Review database query performance")
            print("   - Consider adding caching layers")
        
        avg_rps = statistics.mean([r.requests_per_second for r in self.results])
        if avg_rps < 50:  # Average RPS < 50
            print("   - Consider horizontal scaling")
            print("   - Optimize application performance")
            print("   - Review resource allocation")
        
        if any(r.error_rate > 0 for r in self.results):
            print("   - Investigate error causes")
            print("   - Implement better error handling")
            print("   - Add circuit breakers for resilience")


async def main():
    """Main load testing function"""
    import sys
    
    # Allow testing against different environments
    base_url = BASE_URL
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"Load testing against: {base_url}")
    
    tester = LoadTester(base_url)
    await tester.run_load_tests()


if __name__ == "__main__":
    asyncio.run(main())
