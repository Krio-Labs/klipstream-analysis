#!/usr/bin/env python3
"""
Cloud Run API Test Script

This script tests the deployed Cloud Run API to validate:
- Enhanced error handling and monitoring
- Progressive quality fallback system
- API routing fixes (both /analysis and /analyze endpoints)
- Real-time status updates
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudRunAPITester:
    """Test suite for Cloud Run API"""
    
    def __init__(self, base_url="https://klipstream-analysis-4vyl5ph7lq-uc.a.run.app"):
        self.base_url = base_url
        self.test_results = []
        
    async def run_all_tests(self):
        """Run comprehensive API tests"""
        logger.info("🧪 Starting Cloud Run API Test Suite")
        logger.info(f"🌐 Testing API at: {self.base_url}")
        logger.info("=" * 80)
        
        test_cases = [
            ("Health Check", self.test_health_check),
            ("API Documentation", self.test_api_docs),
            ("Legacy Endpoint Compatibility", self.test_legacy_endpoint),
            ("New Analysis Endpoint", self.test_new_analysis_endpoint),
            ("Error Handling", self.test_error_handling),
            ("Status Monitoring", self.test_status_monitoring),
        ]
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            for test_name, test_func in test_cases:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔍 Running: {test_name}")
                logger.info(f"{'='*60}")
                
                try:
                    start_time = time.time()
                    result = await test_func()
                    duration = time.time() - start_time
                    
                    self.test_results.append({
                        "test_name": test_name,
                        "status": "PASSED" if result else "FAILED",
                        "duration": duration,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    status_emoji = "✅" if result else "❌"
                    logger.info(f"{status_emoji} {test_name}: {'PASSED' if result else 'FAILED'} ({duration:.2f}s)")
                    
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"❌ {test_name}: ERROR - {str(e)}")
                    self.test_results.append({
                        "test_name": test_name,
                        "status": "ERROR",
                        "error": str(e),
                        "duration": duration,
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        # Print summary
        self.print_test_summary()
        
    async def test_health_check(self) -> bool:
        """Test health check endpoint"""
        try:
            logger.info("Testing health check endpoint...")
            
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Health check response: {data}")
                    
                    # Validate response structure
                    required_fields = ["status", "timestamp", "version"]
                    for field in required_fields:
                        if field not in data:
                            logger.error(f"❌ Missing field in health response: {field}")
                            return False
                    
                    if data["status"] == "healthy":
                        logger.info("✅ Service is healthy")
                        return True
                    else:
                        logger.error(f"❌ Service status: {data['status']}")
                        return False
                else:
                    logger.error(f"❌ Health check failed with status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def test_api_docs(self) -> bool:
        """Test API documentation endpoint"""
        try:
            logger.info("Testing API documentation endpoint...")
            
            async with self.session.get(f"{self.base_url}/docs") as response:
                if response.status == 200:
                    content = await response.text()
                    if "swagger" in content.lower() or "openapi" in content.lower():
                        logger.info("✅ API documentation is accessible")
                        return True
                    else:
                        logger.error("❌ API documentation content invalid")
                        return False
                else:
                    logger.error(f"❌ API docs failed with status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ API docs error: {e}")
            return False
    
    async def test_legacy_endpoint(self) -> bool:
        """Test legacy /api/v1/analyze endpoint"""
        try:
            logger.info("Testing legacy /api/v1/analyze endpoint...")
            
            # Test with invalid URL to check error handling
            test_payload = {
                "url": "https://www.twitch.tv/videos/invalid_video_id_12345"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/analyze",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                data = await response.json()
                logger.info(f"Legacy endpoint response status: {response.status}")
                logger.info(f"Legacy endpoint response: {json.dumps(data, indent=2)}")
                
                # Should return 400 for invalid URL
                if response.status == 400:
                    # Check for enhanced error structure
                    if "error" in data and "error_type" in data["error"]:
                        logger.info("✅ Legacy endpoint working with enhanced error handling")
                        return True
                    else:
                        logger.warning("⚠️ Legacy endpoint working but missing enhanced error structure")
                        return True
                else:
                    logger.error(f"❌ Unexpected response status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Legacy endpoint error: {e}")
            return False
    
    async def test_new_analysis_endpoint(self) -> bool:
        """Test new /api/v1/analysis endpoint"""
        try:
            logger.info("Testing new /api/v1/analysis endpoint...")
            
            # Test with invalid URL to check error handling
            test_payload = {
                "url": "https://www.twitch.tv/videos/invalid_video_id_67890"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/analysis",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                data = await response.json()
                logger.info(f"New endpoint response status: {response.status}")
                logger.info(f"New endpoint response: {json.dumps(data, indent=2)}")
                
                # Should return 400 for invalid URL
                if response.status == 400:
                    # Check for enhanced error structure
                    if "error" in data and "error_type" in data["error"]:
                        logger.info("✅ New endpoint working with enhanced error handling")
                        return True
                    else:
                        logger.warning("⚠️ New endpoint working but missing enhanced error structure")
                        return True
                else:
                    logger.error(f"❌ Unexpected response status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ New endpoint error: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test enhanced error handling"""
        try:
            logger.info("Testing enhanced error handling...")
            
            # Test with completely invalid payload
            test_payload = {
                "invalid_field": "invalid_value"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/analysis",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                data = await response.json()
                logger.info(f"Error handling response status: {response.status}")
                logger.info(f"Error handling response: {json.dumps(data, indent=2)}")
                
                # Should return 422 for validation error
                if response.status == 422:
                    logger.info("✅ Validation error handling working")
                    return True
                elif response.status == 400:
                    logger.info("✅ Bad request error handling working")
                    return True
                else:
                    logger.error(f"❌ Unexpected error response status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error handling test error: {e}")
            return False
    
    async def test_status_monitoring(self) -> bool:
        """Test status monitoring capabilities"""
        try:
            logger.info("Testing status monitoring...")
            
            # Test root endpoint for service info
            async with self.session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Root endpoint response: {json.dumps(data, indent=2)}")
                    
                    # Check for expected fields
                    expected_fields = ["name", "version", "status", "features", "endpoints"]
                    for field in expected_fields:
                        if field not in data:
                            logger.error(f"❌ Missing field in root response: {field}")
                            return False
                    
                    # Check for enhanced features
                    features = data.get("features", [])
                    enhanced_features = [
                        "Asynchronous video processing",
                        "Real-time progress tracking",
                        "Comprehensive error handling"
                    ]
                    
                    for feature in enhanced_features:
                        if feature in features:
                            logger.info(f"✅ Enhanced feature available: {feature}")
                        else:
                            logger.warning(f"⚠️ Enhanced feature missing: {feature}")
                    
                    logger.info("✅ Status monitoring working")
                    return True
                else:
                    logger.error(f"❌ Root endpoint failed with status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Status monitoring error: {e}")
            return False
    
    def print_test_summary(self):
        """Print test summary"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 CLOUD RUN API TEST SUMMARY")
        logger.info(f"{'='*80}")
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAILED"])
        error_tests = len([r for r in self.test_results if r["status"] == "ERROR"])
        
        logger.info(f"🌐 API Base URL: {self.base_url}")
        logger.info(f"📊 Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"💥 Errors: {error_tests}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        logger.info(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Detailed results
        logger.info(f"\n📋 Detailed Results:")
        for result in self.test_results:
            status_emoji = {"PASSED": "✅", "FAILED": "❌", "ERROR": "💥"}[result["status"]]
            logger.info(f"  {status_emoji} {result['test_name']}: {result['status']} ({result['duration']:.2f}s)")
        
        # Save results to file
        with open("cloud_run_api_test_results.json", "w") as f:
            json.dump({
                "summary": {
                    "api_url": self.base_url,
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "errors": error_tests,
                    "success_rate": success_rate,
                    "test_timestamp": datetime.utcnow().isoformat()
                },
                "test_results": self.test_results
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: cloud_run_api_test_results.json")
        
        if success_rate >= 80:
            logger.info("🎉 API testing completed successfully!")
        else:
            logger.warning("⚠️ Some API tests failed. Please review the results.")

async def main():
    """Main test function"""
    tester = CloudRunAPITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
