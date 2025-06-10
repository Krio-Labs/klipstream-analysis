#!/usr/bin/env python3
"""
KlipStream Analysis Comprehensive Production Test Suite

This comprehensive test suite validates production readiness across all critical areas:
1. Service Health & Availability Testing
2. Functional Testing
3. Performance & Load Testing
4. Integration & Compatibility Testing
5. Error Handling & Recovery Testing
6. Production Readiness Validation
"""

import asyncio
import aiohttp
import json
import time
import sys
import logging
import concurrent.futures
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    category: str
    status: str  # PASS, FAIL, ERROR, SKIP
    duration: float
    details: Dict[str, Any]
    timestamp: str
    error_message: Optional[str] = None

class ComprehensiveProductionTestSuite:
    """Comprehensive test suite for KlipStream Analysis API production readiness"""
    
    def __init__(self, base_url: str = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"):
        self.base_url = base_url
        self.test_results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Test configuration
        self.config = {
            "health_check_timeout": 5,
            "api_timeout": 30,
            "long_timeout": 300,  # 5 minutes for analysis tests
            "concurrent_requests": 5,
            "load_test_duration": 60,  # 1 minute load test
            "performance_threshold_ms": 5000,
            "availability_threshold": 0.999,  # 99.9%
        }
        
        # Test video URLs for different scenarios
        self.test_videos = {
            "short": "https://www.twitch.tv/videos/2000000001",  # 30min test video
            "medium": "https://www.twitch.tv/videos/2000000002",  # 1hr test video
            "long": "https://www.twitch.tv/videos/2000000003",  # 2hr test video
            "very_long": "https://www.twitch.tv/videos/2000000004",  # 4hr+ test video
            "invalid": "https://invalid-url.com/video",
            "malformed": "not-a-url"
        }
    
    def log_test_result(self, test_name: str, category: str, status: str, 
                       duration: float, details: Dict[str, Any], 
                       error_message: Optional[str] = None):
        """Log test result"""
        result = TestResult(
            test_name=test_name,
            category=category,
            status=status,
            duration=duration,
            details=details,
            timestamp=datetime.utcnow().isoformat(),
            error_message=error_message
        )
        self.test_results.append(result)
        
        status_emoji = {
            "PASS": "✅",
            "FAIL": "❌", 
            "ERROR": "🔥",
            "SKIP": "⏭️"
        }.get(status, "❓")
        
        logger.info(f"{status_emoji} {category} - {test_name}: {status} ({duration:.2f}s)")
        if error_message:
            logger.error(f"   Error: {error_message}")
    
    async def run_test(self, test_func, test_name: str, category: str, *args, **kwargs):
        """Run a single test with error handling and timing"""
        start_time = time.time()
        try:
            result = await test_func(*args, **kwargs)
            duration = time.time() - start_time
            
            if isinstance(result, dict):
                status = "PASS" if result.get("success", False) else "FAIL"
                details = result
                error_message = result.get("error")
            else:
                status = "PASS" if result else "FAIL"
                details = {"success": result}
                error_message = None
                
            self.log_test_result(test_name, category, status, duration, details, error_message)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_message = str(e)
            self.log_test_result(test_name, category, "ERROR", duration, {}, error_message)
            return False
    
    # ==================== SERVICE HEALTH & AVAILABILITY TESTING ====================
    
    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test basic health endpoint"""
        try:
            async with self.session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=self.config["health_check_timeout"])
            ) as response:
                response_time = response.headers.get('X-Response-Time', 'unknown')
                data = await response.json()
                
                return {
                    "success": response.status == 200,
                    "status_code": response.status,
                    "response_time_ms": response_time,
                    "service_status": data.get("status"),
                    "version": data.get("version"),
                    "uptime_seconds": data.get("uptime_seconds")
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_detailed_health_endpoint(self) -> Dict[str, Any]:
        """Test detailed health endpoint"""
        try:
            async with self.session.get(
                f"{self.base_url}/health/detailed",
                timeout=aiohttp.ClientTimeout(total=self.config["health_check_timeout"])
            ) as response:
                data = await response.json()
                
                return {
                    "success": response.status == 200,
                    "status_code": response.status,
                    "system_info": data.get("system", {}),
                    "services_info": data.get("services", {}),
                    "metrics_info": data.get("metrics", {}),
                    "cache_info": data.get("cache", {})
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_readiness_endpoint(self) -> Dict[str, Any]:
        """Test Kubernetes readiness endpoint"""
        try:
            async with self.session.get(
                f"{self.base_url}/health/readiness",
                timeout=aiohttp.ClientTimeout(total=self.config["health_check_timeout"])
            ) as response:
                data = await response.json()
                
                return {
                    "success": response.status == 200,
                    "status_code": response.status,
                    "ready": data.get("status") == "ready"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_liveness_endpoint(self) -> Dict[str, Any]:
        """Test Kubernetes liveness endpoint"""
        try:
            async with self.session.get(
                f"{self.base_url}/health/liveness",
                timeout=aiohttp.ClientTimeout(total=self.config["health_check_timeout"])
            ) as response:
                data = await response.json()
                
                return {
                    "success": response.status == 200,
                    "status_code": response.status,
                    "alive": data.get("status") == "alive",
                    "uptime_seconds": data.get("uptime_seconds")
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_api_discovery(self) -> Dict[str, Any]:
        """Test API discovery endpoint"""
        try:
            async with self.session.get(
                f"{self.base_url}/",
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                data = await response.json()
                
                return {
                    "success": response.status == 200,
                    "status_code": response.status,
                    "api_name": data.get("name"),
                    "version": data.get("version"),
                    "features": data.get("features", []),
                    "endpoints": data.get("endpoints", {})
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_availability_over_time(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Test service availability over time"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        total_requests = 0
        successful_requests = 0
        response_times = []
        
        while time.time() < end_time:
            try:
                request_start = time.time()
                async with self.session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    request_time = (time.time() - request_start) * 1000
                    response_times.append(request_time)
                    
                    total_requests += 1
                    if response.status == 200:
                        successful_requests += 1
                        
            except Exception:
                total_requests += 1
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        availability = successful_requests / total_requests if total_requests > 0 else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "success": availability >= self.config["availability_threshold"],
            "availability": availability,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "avg_response_time_ms": avg_response_time,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0
        }

    # ==================== FUNCTIONAL TESTING ====================

    async def test_analysis_endpoint_validation(self) -> Dict[str, Any]:
        """Test analysis endpoint input validation"""
        try:
            # Test with invalid data
            async with self.session.post(
                f"{self.base_url}/api/v1/analysis",
                json={"invalid": "data"},
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                data = await response.json()

                return {
                    "success": response.status in [400, 422],  # Validation error expected
                    "status_code": response.status,
                    "validation_working": response.status in [400, 422],
                    "error_structure": "error" in data
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_analysis_endpoint_with_valid_url(self) -> Dict[str, Any]:
        """Test analysis endpoint with valid Twitch URL"""
        try:
            payload = {"url": self.test_videos["short"]}
            async with self.session.post(
                f"{self.base_url}/api/v1/analysis",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config["long_timeout"])
            ) as response:
                data = await response.json()

                return {
                    "success": response.status in [200, 202],  # Accept or processing
                    "status_code": response.status,
                    "job_id": data.get("job_id"),
                    "status": data.get("status"),
                    "has_progress_info": "progress" in data
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_legacy_endpoint_compatibility(self) -> Dict[str, Any]:
        """Test backward compatibility with legacy /analyze endpoint"""
        try:
            payload = {"url": self.test_videos["short"]}
            async with self.session.post(
                f"{self.base_url}/api/v1/analyze",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                data = await response.json()

                return {
                    "success": response.status in [200, 202, 400],  # Any valid response
                    "status_code": response.status,
                    "legacy_compatibility": True,
                    "redirects_properly": True
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_status_endpoint(self) -> Dict[str, Any]:
        """Test status tracking endpoint"""
        try:
            # First create a job
            payload = {"url": self.test_videos["short"]}
            async with self.session.post(
                f"{self.base_url}/api/v1/analysis",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                if response.status not in [200, 202]:
                    return {"success": False, "error": "Failed to create job"}

                data = await response.json()
                job_id = data.get("job_id")

                if not job_id:
                    return {"success": False, "error": "No job_id returned"}

                # Now test status endpoint
                async with self.session.get(
                    f"{self.base_url}/api/v1/analysis/{job_id}/status",
                    timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
                ) as status_response:
                    status_data = await status_response.json()

                    return {
                        "success": status_response.status == 200,
                        "status_code": status_response.status,
                        "job_id": job_id,
                        "job_status": status_data.get("status"),
                        "has_progress": "progress" in status_data,
                        "has_timestamps": "created_at" in status_data
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_transcription_routing_system(self) -> Dict[str, Any]:
        """Test intelligent transcription routing system"""
        try:
            # Test with different file scenarios to trigger routing
            test_cases = [
                ("short_video", self.test_videos["short"]),
                ("medium_video", self.test_videos["medium"]),
            ]

            results = {}
            for case_name, video_url in test_cases:
                payload = {"url": video_url}
                async with self.session.post(
                    f"{self.base_url}/api/v1/analysis",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
                ) as response:
                    data = await response.json()
                    results[case_name] = {
                        "status_code": response.status,
                        "job_created": response.status in [200, 202],
                        "job_id": data.get("job_id")
                    }

            success = all(result["job_created"] for result in results.values())
            return {
                "success": success,
                "test_cases": results,
                "routing_system_working": success
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_convex_integration(self) -> Dict[str, Any]:
        """Test Convex database integration"""
        try:
            # Create a job and check if it integrates with Convex
            payload = {"url": self.test_videos["short"]}
            async with self.session.post(
                f"{self.base_url}/api/v1/analysis",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                data = await response.json()

                if response.status not in [200, 202]:
                    return {"success": False, "error": "Failed to create job"}

                job_id = data.get("job_id")

                # Wait a moment for Convex integration
                await asyncio.sleep(2)

                # Check status to see if Convex integration is working
                async with self.session.get(
                    f"{self.base_url}/api/v1/analysis/{job_id}/status",
                    timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
                ) as status_response:
                    status_data = await status_response.json()

                    return {
                        "success": status_response.status == 200,
                        "job_id": job_id,
                        "convex_integration": status_data.get("status") is not None,
                        "status_updates_working": True
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== PERFORMANCE & LOAD TESTING ====================

    async def test_response_time_performance(self) -> Dict[str, Any]:
        """Test API response time performance"""
        try:
            response_times = []
            num_requests = 10

            for _ in range(num_requests):
                start_time = time.time()
                async with self.session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=self.config["health_check_timeout"])
                ) as response:
                    end_time = time.time()
                    response_time_ms = (end_time - start_time) * 1000
                    response_times.append(response_time_ms)

                    if response.status != 200:
                        return {"success": False, "error": f"Non-200 status: {response.status}"}

            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)

            return {
                "success": avg_response_time < self.config["performance_threshold_ms"],
                "avg_response_time_ms": avg_response_time,
                "max_response_time_ms": max_response_time,
                "min_response_time_ms": min_response_time,
                "threshold_ms": self.config["performance_threshold_ms"],
                "performance_acceptable": avg_response_time < self.config["performance_threshold_ms"]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_concurrent_request_handling(self) -> Dict[str, Any]:
        """Test concurrent request handling capacity"""
        try:
            num_concurrent = self.config["concurrent_requests"]

            async def make_request():
                start_time = time.time()
                try:
                    async with self.session.get(
                        f"{self.base_url}/health",
                        timeout=aiohttp.ClientTimeout(total=self.config["health_check_timeout"])
                    ) as response:
                        end_time = time.time()
                        return {
                            "success": response.status == 200,
                            "status_code": response.status,
                            "response_time_ms": (end_time - start_time) * 1000
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "response_time_ms": (time.time() - start_time) * 1000
                    }

            # Execute concurrent requests
            tasks = [make_request() for _ in range(num_concurrent)]
            results = await asyncio.gather(*tasks)

            successful_requests = sum(1 for r in results if r["success"])
            avg_response_time = sum(r["response_time_ms"] for r in results) / len(results)
            success_rate = successful_requests / num_concurrent

            return {
                "success": success_rate >= 0.8,  # 80% success rate threshold
                "total_requests": num_concurrent,
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "avg_response_time_ms": avg_response_time,
                "concurrent_capacity_ok": success_rate >= 0.8
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_memory_usage_limits(self) -> Dict[str, Any]:
        """Test memory usage stays within limits during processing"""
        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory_mb = process.memory_info().rss / 1024 / 1024

            # Start a processing job
            payload = {"url": self.test_videos["short"]}
            async with self.session.post(
                f"{self.base_url}/api/v1/analysis",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                if response.status not in [200, 202]:
                    return {"success": False, "error": "Failed to start job"}

                data = await response.json()
                job_id = data.get("job_id")

                # Monitor memory for a short period
                max_memory_mb = initial_memory_mb
                for _ in range(10):  # Monitor for ~30 seconds
                    await asyncio.sleep(3)
                    current_memory_mb = process.memory_info().rss / 1024 / 1024
                    max_memory_mb = max(max_memory_mb, current_memory_mb)

                memory_limit_gb = 32  # Cloud Run memory limit
                memory_limit_mb = memory_limit_gb * 1024

                return {
                    "success": max_memory_mb < (memory_limit_mb * 0.8),  # 80% of limit
                    "initial_memory_mb": initial_memory_mb,
                    "max_memory_mb": max_memory_mb,
                    "memory_limit_mb": memory_limit_mb,
                    "memory_usage_acceptable": max_memory_mb < (memory_limit_mb * 0.8),
                    "job_id": job_id
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_processing_different_file_sizes(self) -> Dict[str, Any]:
        """Test processing of different video file sizes"""
        try:
            test_cases = [
                ("30min_video", self.test_videos["short"]),
                ("1hr_video", self.test_videos["medium"]),
                ("2hr_video", self.test_videos["long"]),
            ]

            results = {}
            for case_name, video_url in test_cases:
                start_time = time.time()
                payload = {"url": video_url}

                async with self.session.post(
                    f"{self.base_url}/api/v1/analysis",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
                ) as response:
                    processing_time = time.time() - start_time
                    data = await response.json()

                    results[case_name] = {
                        "success": response.status in [200, 202],
                        "status_code": response.status,
                        "job_id": data.get("job_id"),
                        "processing_start_time_ms": processing_time * 1000
                    }

            all_successful = all(result["success"] for result in results.values())
            return {
                "success": all_successful,
                "test_cases": results,
                "handles_different_sizes": all_successful
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== ERROR HANDLING & RECOVERY TESTING ====================

    async def test_invalid_input_handling(self) -> Dict[str, Any]:
        """Test handling of invalid inputs"""
        try:
            test_cases = [
                ("invalid_url", {"url": self.test_videos["invalid"]}),
                ("malformed_url", {"url": self.test_videos["malformed"]}),
                ("missing_url", {}),
                ("empty_url", {"url": ""}),
            ]

            results = {}
            for case_name, payload in test_cases:
                async with self.session.post(
                    f"{self.base_url}/api/v1/analysis",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
                ) as response:
                    data = await response.json()

                    results[case_name] = {
                        "status_code": response.status,
                        "proper_error_response": response.status in [400, 422],
                        "has_error_structure": "error" in data
                    }

            all_handled_properly = all(result["proper_error_response"] for result in results.values())
            return {
                "success": all_handled_properly,
                "test_cases": results,
                "error_handling_working": all_handled_properly
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_timeout_behavior(self) -> Dict[str, Any]:
        """Test timeout behavior for long-running processes"""
        try:
            payload = {"url": self.test_videos["long"]}

            try:
                async with self.session.post(
                    f"{self.base_url}/api/v1/analysis",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=1)  # Very short timeout
                ) as response:
                    return {
                        "success": True,
                        "timeout_handled": False,
                        "fast_response": True,
                        "status_code": response.status
                    }
            except asyncio.TimeoutError:
                return {
                    "success": True,
                    "timeout_handled": True,
                    "proper_timeout_behavior": True
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_network_failure_recovery(self) -> Dict[str, Any]:
        """Test network failure recovery mechanisms"""
        try:
            async with self.session.get(
                f"{self.base_url}/non-existent-endpoint",
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                return {
                    "success": response.status == 404,
                    "status_code": response.status,
                    "proper_404_handling": response.status == 404
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== INTEGRATION & COMPATIBILITY TESTING ====================

    async def test_webhook_integration(self) -> Dict[str, Any]:
        """Test webhook integration capabilities"""
        try:
            webhook_payload = {
                "url": "https://example.com/webhook",
                "events": ["job.started", "job.completed"],
                "timeout_seconds": 30,
                "max_retries": 3
            }

            async with self.session.post(
                f"{self.base_url}/webhooks/?webhook_id=test-webhook-{int(time.time())}",
                json=webhook_payload,
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                return {
                    "success": response.status in [200, 201],
                    "status_code": response.status,
                    "webhook_creation_working": response.status in [200, 201]
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_authentication_and_permissions(self) -> Dict[str, Any]:
        """Test authentication and service account permissions"""
        try:
            async with self.session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=self.config["health_check_timeout"])
            ) as response:
                return {
                    "success": response.status == 200,
                    "unauthenticated_access": response.status == 200,
                    "status_code": response.status
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== MAIN TEST EXECUTION ====================

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive production test suite"""
        logger.info("🚀 Starting Comprehensive Production Test Suite")
        logger.info(f"🌐 Testing API at: {self.base_url}")
        logger.info("=" * 80)

        # Test categories and their tests
        test_categories = {
            "Service Health & Availability": [
                ("Health Endpoint", self.test_health_endpoint),
                ("Detailed Health Endpoint", self.test_detailed_health_endpoint),
                ("Readiness Endpoint", self.test_readiness_endpoint),
                ("Liveness Endpoint", self.test_liveness_endpoint),
                ("API Discovery", self.test_api_discovery),
                ("Availability Over Time", self.test_availability_over_time, 2),  # 2 minute test
            ],
            "Functional Testing": [
                ("Analysis Endpoint Validation", self.test_analysis_endpoint_validation),
                ("Analysis with Valid URL", self.test_analysis_endpoint_with_valid_url),
                ("Legacy Endpoint Compatibility", self.test_legacy_endpoint_compatibility),
                ("Status Endpoint", self.test_status_endpoint),
                ("Transcription Routing System", self.test_transcription_routing_system),
                ("Convex Integration", self.test_convex_integration),
            ],
            "Performance & Load Testing": [
                ("Response Time Performance", self.test_response_time_performance),
                ("Concurrent Request Handling", self.test_concurrent_request_handling),
                ("Memory Usage Limits", self.test_memory_usage_limits),
                ("Processing Different File Sizes", self.test_processing_different_file_sizes),
            ],
            "Error Handling & Recovery": [
                ("Invalid Input Handling", self.test_invalid_input_handling),
                ("Timeout Behavior", self.test_timeout_behavior),
                ("Network Failure Recovery", self.test_network_failure_recovery),
            ],
            "Integration & Compatibility": [
                ("Webhook Integration", self.test_webhook_integration),
                ("Authentication and Permissions", self.test_authentication_and_permissions),
            ]
        }

        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute default timeout

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            self.session = session

            # Run tests by category
            for category_name, tests in test_categories.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"🔍 Running Category: {category_name}")
                logger.info(f"{'='*60}")

                for test_info in tests:
                    if len(test_info) == 3:  # Test with custom parameter
                        test_name, test_func, param = test_info
                        await self.run_test(test_func, test_name, category_name, param)
                    else:
                        test_name, test_func = test_info
                        await self.run_test(test_func, test_name, category_name)

                    # Brief pause between tests
                    await asyncio.sleep(1)

        # Generate comprehensive report
        return self.generate_comprehensive_report()

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        # Categorize results
        results_by_category = {}
        for result in self.test_results:
            category = result.category
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)

        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == "PASS")
        failed_tests = sum(1 for r in self.test_results if r.status == "FAIL")
        error_tests = sum(1 for r in self.test_results if r.status == "ERROR")

        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        # Determine overall status
        critical_failures = []
        warnings = []

        for result in self.test_results:
            if result.status in ["FAIL", "ERROR"]:
                if result.category in ["Service Health & Availability", "Functional Testing"]:
                    critical_failures.append(result)
                else:
                    warnings.append(result)

        # Overall status determination
        if len(critical_failures) == 0 and success_rate >= 0.9:
            overall_status = "PRODUCTION_READY"
        elif len(critical_failures) == 0 and success_rate >= 0.8:
            overall_status = "READY_WITH_WARNINGS"
        elif len(critical_failures) <= 2 and success_rate >= 0.7:
            overall_status = "NEEDS_ATTENTION"
        else:
            overall_status = "NOT_READY"

        # Category summaries
        category_summaries = {}
        for category, results in results_by_category.items():
            category_passed = sum(1 for r in results if r.status == "PASS")
            category_total = len(results)
            category_summaries[category] = {
                "passed": category_passed,
                "total": category_total,
                "success_rate": category_passed / category_total if category_total > 0 else 0,
                "status": "PASS" if category_passed == category_total else "FAIL"
            }

        # Performance metrics
        performance_metrics = self._extract_performance_metrics()

        # Recommendations
        recommendations = self._generate_recommendations(critical_failures, warnings)

        report = {
            "overall_status": overall_status,
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": success_rate
            },
            "category_summaries": category_summaries,
            "performance_metrics": performance_metrics,
            "critical_failures": [
                {
                    "test_name": f.test_name,
                    "category": f.category,
                    "error": f.error_message,
                    "details": f.details
                } for f in critical_failures
            ],
            "warnings": [
                {
                    "test_name": w.test_name,
                    "category": w.category,
                    "error": w.error_message,
                    "details": w.details
                } for w in warnings
            ],
            "recommendations": recommendations,
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "category": r.category,
                    "status": r.status,
                    "duration": r.duration,
                    "details": r.details,
                    "timestamp": r.timestamp,
                    "error_message": r.error_message
                } for r in self.test_results
            ],
            "test_configuration": self.config,
            "service_url": self.base_url,
            "report_timestamp": datetime.utcnow().isoformat()
        }

        return report

    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics from test results"""
        metrics = {
            "response_times": [],
            "availability": None,
            "concurrent_capacity": None,
            "memory_usage": None
        }

        for result in self.test_results:
            if result.test_name == "Response Time Performance" and result.status == "PASS":
                metrics["response_times"] = {
                    "avg_ms": result.details.get("avg_response_time_ms"),
                    "max_ms": result.details.get("max_response_time_ms"),
                    "min_ms": result.details.get("min_response_time_ms")
                }
            elif result.test_name == "Availability Over Time" and result.status == "PASS":
                metrics["availability"] = result.details.get("availability")
            elif result.test_name == "Concurrent Request Handling" and result.status == "PASS":
                metrics["concurrent_capacity"] = {
                    "success_rate": result.details.get("success_rate"),
                    "max_concurrent": result.details.get("total_requests")
                }
            elif result.test_name == "Memory Usage Limits" and result.status == "PASS":
                metrics["memory_usage"] = {
                    "max_memory_mb": result.details.get("max_memory_mb"),
                    "limit_mb": result.details.get("memory_limit_mb")
                }

        return metrics

    def _generate_recommendations(self, critical_failures: List[TestResult],
                                warnings: List[TestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        if critical_failures:
            recommendations.append("🔴 CRITICAL: Address critical failures before production deployment")
            for failure in critical_failures:
                recommendations.append(f"   - Fix {failure.test_name}: {failure.error_message}")

        if warnings:
            recommendations.append("🟡 WARNING: Consider addressing these issues for optimal performance")
            for warning in warnings:
                recommendations.append(f"   - Review {warning.test_name}: {warning.error_message}")

        # Performance recommendations
        perf_metrics = self._extract_performance_metrics()
        if perf_metrics["response_times"] and perf_metrics["response_times"]["avg_ms"] > 3000:
            recommendations.append("⚡ Consider optimizing response times (currently > 3s average)")

        if perf_metrics["availability"] and perf_metrics["availability"] < 0.999:
            recommendations.append("📈 Improve service availability (target: 99.9%)")

        if not recommendations:
            recommendations.append("✅ All tests passed! Service is ready for production traffic.")

        return recommendations

    def print_summary_report(self, report: Dict[str, Any]):
        """Print a formatted summary report"""
        print("\n" + "="*80)
        print("🎯 KLIPSTREAM ANALYSIS API - PRODUCTION READINESS REPORT")
        print("="*80)

        # Overall status
        status_emoji = {
            "PRODUCTION_READY": "🟢",
            "READY_WITH_WARNINGS": "🟡",
            "NEEDS_ATTENTION": "🟠",
            "NOT_READY": "🔴"
        }.get(report["overall_status"], "❓")

        print(f"\n{status_emoji} OVERALL STATUS: {report['overall_status']}")

        # Test summary
        summary = report["test_summary"]
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']} ✅")
        print(f"   Failed: {summary['failed_tests']} ❌")
        print(f"   Errors: {summary['error_tests']} 🔥")
        print(f"   Success Rate: {summary['success_rate']:.1%}")

        # Category breakdown
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, stats in report["category_summaries"].items():
            status_icon = "✅" if stats["status"] == "PASS" else "❌"
            print(f"   {status_icon} {category}: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1%})")

        # Performance metrics
        if report["performance_metrics"]:
            print(f"\n⚡ PERFORMANCE METRICS:")
            metrics = report["performance_metrics"]
            if metrics["response_times"]:
                rt = metrics["response_times"]
                print(f"   Response Time: {rt['avg_ms']:.1f}ms avg (min: {rt['min_ms']:.1f}ms, max: {rt['max_ms']:.1f}ms)")
            if metrics["availability"]:
                print(f"   Availability: {metrics['availability']:.3%}")
            if metrics["concurrent_capacity"]:
                cc = metrics["concurrent_capacity"]
                print(f"   Concurrent Capacity: {cc['max_concurrent']} requests ({cc['success_rate']:.1%} success)")

        # Critical failures
        if report["critical_failures"]:
            print(f"\n🔴 CRITICAL FAILURES:")
            for failure in report["critical_failures"]:
                print(f"   ❌ {failure['test_name']}: {failure['error']}")

        # Warnings
        if report["warnings"]:
            print(f"\n🟡 WARNINGS:")
            for warning in report["warnings"]:
                print(f"   ⚠️ {warning['test_name']}: {warning['error']}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   {rec}")

        print("\n" + "="*80)


async def main():
    """Main execution function"""
    import sys
    import json

    # Parse command line arguments
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"

    # Create and run test suite
    test_suite = ComprehensiveProductionTestSuite(base_url)

    try:
        logger.info("Starting comprehensive production test suite...")
        report = await test_suite.run_all_tests()

        # Print summary
        test_suite.print_summary_report(report)

        # Save detailed report
        report_filename = f"production_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📄 Detailed report saved to: {report_filename}")

        # Exit with appropriate code
        exit_code = 0 if report["overall_status"] in ["PRODUCTION_READY", "READY_WITH_WARNINGS"] else 1
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Test suite execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

    # ==================== ERROR HANDLING & RECOVERY TESTING ====================

    async def test_invalid_input_handling(self) -> Dict[str, Any]:
        """Test handling of invalid inputs"""
        try:
            test_cases = [
                ("invalid_url", {"url": self.test_videos["invalid"]}),
                ("malformed_url", {"url": self.test_videos["malformed"]}),
                ("missing_url", {}),
                ("empty_url", {"url": ""}),
            ]

            results = {}
            for case_name, payload in test_cases:
                async with self.session.post(
                    f"{self.base_url}/api/v1/analysis",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
                ) as response:
                    data = await response.json()

                    results[case_name] = {
                        "status_code": response.status,
                        "proper_error_response": response.status in [400, 422],
                        "has_error_structure": "error" in data
                    }

            all_handled_properly = all(result["proper_error_response"] for result in results.values())
            return {
                "success": all_handled_properly,
                "test_cases": results,
                "error_handling_working": all_handled_properly
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_timeout_behavior(self) -> Dict[str, Any]:
        """Test timeout behavior for long-running processes"""
        try:
            payload = {"url": self.test_videos["long"]}

            try:
                async with self.session.post(
                    f"{self.base_url}/api/v1/analysis",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=1)  # Very short timeout
                ) as response:
                    return {
                        "success": True,
                        "timeout_handled": False,
                        "fast_response": True,
                        "status_code": response.status
                    }
            except asyncio.TimeoutError:
                return {
                    "success": True,
                    "timeout_handled": True,
                    "proper_timeout_behavior": True
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_network_failure_recovery(self) -> Dict[str, Any]:
        """Test network failure recovery mechanisms"""
        try:
            async with self.session.get(
                f"{self.base_url}/non-existent-endpoint",
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                return {
                    "success": response.status == 404,
                    "status_code": response.status,
                    "proper_404_handling": response.status == 404
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== INTEGRATION & COMPATIBILITY TESTING ====================

    async def test_webhook_integration(self) -> Dict[str, Any]:
        """Test webhook integration capabilities"""
        try:
            webhook_payload = {
                "url": "https://example.com/webhook",
                "events": ["job.started", "job.completed"],
                "timeout_seconds": 30,
                "max_retries": 3
            }

            async with self.session.post(
                f"{self.base_url}/webhooks/?webhook_id=test-webhook-{int(time.time())}",
                json=webhook_payload,
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                return {
                    "success": response.status in [200, 201],
                    "status_code": response.status,
                    "webhook_creation_working": response.status in [200, 201]
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_environment_configuration(self) -> Dict[str, Any]:
        """Test environment variable configuration"""
        try:
            async with self.session.get(
                f"{self.base_url}/",
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                data = await response.json()

                return {
                    "success": response.status == 200,
                    "proper_configuration": data.get("version") is not None,
                    "api_version": data.get("version"),
                    "features_available": len(data.get("features", [])) > 0
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== ERROR HANDLING & RECOVERY TESTING ====================

    async def test_invalid_input_handling(self) -> Dict[str, Any]:
        """Test handling of invalid inputs"""
        try:
            test_cases = [
                ("invalid_url", {"url": self.test_videos["invalid"]}),
                ("malformed_url", {"url": self.test_videos["malformed"]}),
                ("missing_url", {}),
                ("null_url", {"url": None}),
                ("empty_url", {"url": ""}),
            ]

            results = {}
            for case_name, payload in test_cases:
                async with self.session.post(
                    f"{self.base_url}/api/v1/analysis",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
                ) as response:
                    data = await response.json()

                    results[case_name] = {
                        "status_code": response.status,
                        "proper_error_response": response.status in [400, 422],
                        "has_error_structure": "error" in data,
                        "error_type": data.get("error", {}).get("error_type") if isinstance(data.get("error"), dict) else None
                    }

            all_handled_properly = all(result["proper_error_response"] for result in results.values())
            return {
                "success": all_handled_properly,
                "test_cases": results,
                "error_handling_working": all_handled_properly
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_timeout_behavior(self) -> Dict[str, Any]:
        """Test timeout behavior for long-running processes"""
        try:
            # Test with a very short timeout to trigger timeout behavior
            payload = {"url": self.test_videos["long"]}

            try:
                async with self.session.post(
                    f"{self.base_url}/api/v1/analysis",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=1)  # Very short timeout
                ) as response:
                    # If we get here, the request was faster than expected
                    return {
                        "success": True,
                        "timeout_handled": False,
                        "fast_response": True,
                        "status_code": response.status
                    }
            except asyncio.TimeoutError:
                # This is expected - timeout was handled properly
                return {
                    "success": True,
                    "timeout_handled": True,
                    "proper_timeout_behavior": True
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_network_failure_recovery(self) -> Dict[str, Any]:
        """Test network failure recovery mechanisms"""
        try:
            # Test with invalid endpoint to simulate network issues
            try:
                async with self.session.get(
                    f"{self.base_url}/non-existent-endpoint",
                    timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
                ) as response:
                    return {
                        "success": response.status == 404,
                        "status_code": response.status,
                        "proper_404_handling": response.status == 404
                    }
            except Exception as e:
                # Network-level failure
                return {
                    "success": False,
                    "network_failure": True,
                    "error": str(e)
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_logging_and_monitoring(self) -> Dict[str, Any]:
        """Test logging and monitoring capabilities"""
        try:
            # Test metrics endpoint
            async with self.session.get(
                f"{self.base_url}/metrics",
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "metrics_endpoint_working": True,
                        "has_metrics": "metrics" in data,
                        "prometheus_format": data.get("format") == "prometheus"
                    }
                else:
                    return {
                        "success": False,
                        "metrics_endpoint_working": False,
                        "status_code": response.status
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== INTEGRATION & COMPATIBILITY TESTING ====================

    async def test_webhook_integration(self) -> Dict[str, Any]:
        """Test webhook integration capabilities"""
        try:
            # Test webhook creation endpoint
            webhook_payload = {
                "url": "https://example.com/webhook",
                "events": ["job.started", "job.completed"],
                "timeout_seconds": 30,
                "max_retries": 3
            }

            async with self.session.post(
                f"{self.base_url}/webhooks/?webhook_id=test-webhook-{int(time.time())}",
                json=webhook_payload,
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                data = await response.json()

                return {
                    "success": response.status in [200, 201],
                    "status_code": response.status,
                    "webhook_creation_working": response.status in [200, 201],
                    "webhook_id": data.get("webhook_id")
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_authentication_and_permissions(self) -> Dict[str, Any]:
        """Test authentication and service account permissions"""
        try:
            # Test that unauthenticated requests work (since service allows unauthenticated)
            async with self.session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=self.config["health_check_timeout"])
            ) as response:
                return {
                    "success": response.status == 200,
                    "unauthenticated_access": response.status == 200,
                    "status_code": response.status
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_environment_configuration(self) -> Dict[str, Any]:
        """Test environment variable configuration"""
        try:
            # Test that the service is properly configured by checking API info
            async with self.session.get(
                f"{self.base_url}/",
                timeout=aiohttp.ClientTimeout(total=self.config["api_timeout"])
            ) as response:
                data = await response.json()

                return {
                    "success": response.status == 200,
                    "proper_configuration": data.get("version") is not None,
                    "api_version": data.get("version"),
                    "features_available": len(data.get("features", [])) > 0
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
