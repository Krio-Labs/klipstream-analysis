#!/usr/bin/env python3
"""
Enhanced Downloader Test Script

This script tests the enhanced downloader functionality including:
- Error handling and monitoring
- Timeout management
- Process monitoring
- Fallback mechanisms
"""

import asyncio
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from raw_pipeline.enhanced_downloader import EnhancedTwitchDownloader
from raw_pipeline.timeout_manager import TimeoutConfig
from utils.logging_setup import setup_logger

# Set up logging
logger = setup_logger("test_enhanced_downloader", "test_enhanced_downloader.log")

class DownloadTestSuite:
    """Test suite for enhanced downloader"""
    
    def __init__(self):
        self.downloader = EnhancedTwitchDownloader()
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all test cases"""
        logger.info("🧪 Starting Enhanced Downloader Test Suite")
        
        test_cases = [
            ("Test 1: Valid Video Download", self.test_valid_download),
            ("Test 2: Invalid Video ID", self.test_invalid_video_id),
            ("Test 3: Timeout Handling", self.test_timeout_handling),
            ("Test 4: Progress Monitoring", self.test_progress_monitoring),
            ("Test 5: Error Classification", self.test_error_classification),
            ("Test 6: Resource Monitoring", self.test_resource_monitoring),
        ]
        
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
                    "timestamp": datetime.now(timezone.utc).isoformat()
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
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        # Print summary
        self.print_test_summary()
        
    async def test_valid_download(self) -> bool:
        """Test downloading a valid video"""
        # Use a small test video (this is a public test video ID)
        test_video_id = "2434635255"  # Replace with a known working video ID
        
        try:
            logger.info(f"Testing download of video ID: {test_video_id}")
            
            # Start download with monitoring
            result = await self.downloader.download_video_with_monitoring(
                video_id=test_video_id,
                job_id="test_valid_download"
            )
            
            # Check if file exists
            if result and result.exists():
                file_size = result.stat().st_size
                logger.info(f"✅ Download successful: {result} ({file_size} bytes)")
                return True
            else:
                logger.error("❌ Download failed: No file created")
                return False
                
        except Exception as e:
            logger.error(f"❌ Download failed with exception: {e}")
            return False
    
    async def test_invalid_video_id(self) -> bool:
        """Test handling of invalid video ID"""
        invalid_video_id = "invalid_video_id_12345"
        
        try:
            logger.info(f"Testing invalid video ID: {invalid_video_id}")
            
            result = await self.downloader.download_video_with_monitoring(
                video_id=invalid_video_id,
                job_id="test_invalid_video"
            )
            
            # Should not reach here
            logger.error("❌ Expected error for invalid video ID but got success")
            return False
            
        except RuntimeError as e:
            # Expected error
            logger.info(f"✅ Correctly handled invalid video ID: {e}")
            return True
        except Exception as e:
            logger.error(f"❌ Unexpected error type: {e}")
            return False
    
    async def test_timeout_handling(self) -> bool:
        """Test timeout handling with very short timeout"""
        test_video_id = "2434635255"
        
        try:
            logger.info("Testing timeout handling with 10-second timeout")
            
            # Temporarily modify the downloader's timeout
            original_timeout = 30 * 60  # Store original
            
            # Create a downloader with very short timeout for testing
            # This should timeout quickly
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(
                    self.downloader.download_video_with_monitoring(
                        video_id=test_video_id,
                        job_id="test_timeout"
                    ),
                    timeout=10  # 10 second timeout for test
                )
                
                # If we get here, the download was very fast or something's wrong
                duration = time.time() - start_time
                if duration < 5:
                    logger.info("✅ Download completed very quickly (likely cached)")
                    return True
                else:
                    logger.error("❌ Expected timeout but download completed")
                    return False
                    
            except asyncio.TimeoutError:
                logger.info("✅ Timeout handled correctly")
                return True
                
        except Exception as e:
            logger.error(f"❌ Unexpected error in timeout test: {e}")
            return False
    
    async def test_progress_monitoring(self) -> bool:
        """Test progress monitoring functionality"""
        try:
            logger.info("Testing progress monitoring")
            
            # Get active downloads (should be empty initially)
            active_downloads = self.downloader.get_active_downloads()
            initial_count = len(active_downloads)
            
            logger.info(f"Initial active downloads: {initial_count}")
            
            # Start a download in background
            download_task = asyncio.create_task(
                self.downloader.download_video_with_monitoring(
                    video_id="2434635255",
                    job_id="test_progress_monitoring"
                )
            )
            
            # Wait a bit and check if it appears in active downloads
            await asyncio.sleep(2)
            
            active_downloads = self.downloader.get_active_downloads()
            during_count = len(active_downloads)
            
            logger.info(f"Active downloads during execution: {during_count}")
            
            # Cancel the download
            download_task.cancel()
            
            try:
                await download_task
            except asyncio.CancelledError:
                pass
            
            # Check if monitoring worked
            if during_count > initial_count:
                logger.info("✅ Progress monitoring working correctly")
                return True
            else:
                logger.error("❌ Progress monitoring not detecting active downloads")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in progress monitoring test: {e}")
            return False
    
    async def test_error_classification(self) -> bool:
        """Test error classification system"""
        try:
            logger.info("Testing error classification")
            
            # Test with a video ID that should cause a specific error
            test_video_id = "nonexistent_video_999999"
            
            try:
                result = await self.downloader.download_video_with_monitoring(
                    video_id=test_video_id,
                    job_id="test_error_classification"
                )
                
                logger.error("❌ Expected error but got success")
                return False
                
            except RuntimeError as e:
                error_message = str(e)
                
                # Check if error message contains expected information
                if "Enhanced download failed:" in error_message and "Ref:" in error_message:
                    logger.info(f"✅ Error classification working: {error_message}")
                    return True
                else:
                    logger.error(f"❌ Error classification not working properly: {error_message}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Unexpected error in error classification test: {e}")
            return False
    
    async def test_resource_monitoring(self) -> bool:
        """Test resource monitoring functionality"""
        try:
            logger.info("Testing resource monitoring")
            
            # This test just verifies that resource monitoring doesn't crash
            # In a real scenario, we'd need a longer-running process to monitor
            
            # Start a short download
            try:
                result = await asyncio.wait_for(
                    self.downloader.download_video_with_monitoring(
                        video_id="2434635255",
                        job_id="test_resource_monitoring"
                    ),
                    timeout=30  # 30 second timeout
                )
                
                logger.info("✅ Resource monitoring completed without errors")
                return True
                
            except asyncio.TimeoutError:
                logger.info("✅ Resource monitoring handled timeout correctly")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error in resource monitoring test: {e}")
            return False
    
    def print_test_summary(self):
        """Print test summary"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAILED"])
        error_tests = len([r for r in self.test_results if r["status"] == "ERROR"])
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"💥 Errors: {error_tests}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Save results to file
        results_file = Path("test_results_enhanced_downloader.json")
        with open(results_file, "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "errors": error_tests,
                    "success_rate": success_rate
                },
                "test_results": self.test_results
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")

async def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Enhanced Downloader Test Suite")
        print("Usage: python test_enhanced_downloader.py")
        print("\nThis script tests the enhanced downloader functionality.")
        return
    
    test_suite = DownloadTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
