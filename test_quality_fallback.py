#!/usr/bin/env python3
"""
Quality Fallback Test Script

This script tests the progressive quality fallback system for handling
memory and timeout issues during video downloads.
"""

import asyncio
import sys
import logging
import json
import time
import psutil
from pathlib import Path
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from raw_pipeline.enhanced_downloader import EnhancedTwitchDownloader
from utils.logging_setup import setup_logger

# Set up logging
logger = setup_logger("test_quality_fallback", "test_quality_fallback.log")

class QualityFallbackTestSuite:
    """Test suite for quality fallback functionality"""
    
    def __init__(self):
        self.downloader = EnhancedTwitchDownloader()
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all quality fallback test cases"""
        logger.info("🧪 Starting Quality Fallback Test Suite")
        
        test_cases = [
            ("Test 1: Quality Recommendation", self.test_quality_recommendation),
            ("Test 2: Thread Optimization", self.test_thread_optimization),
            ("Test 3: Progressive Fallback Simulation", self.test_progressive_fallback_simulation),
            ("Test 4: Memory Constraint Handling", self.test_memory_constraint_handling),
            ("Test 5: Quality Configuration", self.test_quality_configuration),
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
        
    async def test_quality_recommendation(self) -> bool:
        """Test quality recommendation based on available memory"""
        try:
            logger.info("Testing quality recommendation algorithm...")
            
            # Test different memory scenarios
            test_cases = [
                (8192, None, "720p"),  # 8GB, no duration
                (4096, None, "480p"),  # 4GB, no duration
                (2048, None, "360p"),  # 2GB, no duration
                (1024, None, "worst"), # 1GB, no duration
                (8192, 300, "480p"),   # 8GB, 5 hour video (should downgrade)
                (3072, 180, "360p"),   # 3GB, 3 hour video (should downgrade)
            ]
            
            for memory_mb, duration_min, expected_quality in test_cases:
                recommended = self.downloader.get_quality_recommendation(memory_mb, duration_min)
                
                if recommended == expected_quality:
                    logger.info(f"✅ {memory_mb}MB, {duration_min}min -> {recommended} (expected {expected_quality})")
                else:
                    logger.error(f"❌ {memory_mb}MB, {duration_min}min -> {recommended} (expected {expected_quality})")
                    return False
            
            logger.info("✅ All quality recommendation tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quality recommendation test failed: {e}")
            return False
    
    async def test_thread_optimization(self) -> bool:
        """Test thread count optimization based on quality and memory"""
        try:
            logger.info("Testing thread optimization...")
            
            test_cases = [
                ("720p", 8192, 8),  # High quality, high memory
                ("720p", 1024, 2),  # High quality, low memory
                ("480p", 4096, 6),  # Medium quality, medium memory
                ("360p", 2048, 4),  # Low quality, low memory
                ("worst", 1024, 2), # Worst quality, low memory
            ]
            
            for quality, memory_mb, expected_threads in test_cases:
                threads = self.downloader._get_optimal_threads(quality, memory_mb)
                
                if threads == expected_threads:
                    logger.info(f"✅ {quality}, {memory_mb}MB -> {threads} threads (expected {expected_threads})")
                else:
                    logger.error(f"❌ {quality}, {memory_mb}MB -> {threads} threads (expected {expected_threads})")
                    return False
            
            logger.info("✅ All thread optimization tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Thread optimization test failed: {e}")
            return False
    
    async def test_progressive_fallback_simulation(self) -> bool:
        """Test progressive fallback logic without actual downloads"""
        try:
            logger.info("Testing progressive fallback simulation...")
            
            # Check that quality levels are properly configured
            quality_levels = self.downloader.quality_levels
            
            if len(quality_levels) < 2:
                logger.error("❌ Not enough quality levels configured")
                return False
            
            # Verify quality levels are in descending order of resource usage
            expected_qualities = ["720p", "480p", "360p", "worst"]
            actual_qualities = [level["quality"] for level in quality_levels]
            
            for i, expected in enumerate(expected_qualities):
                if i < len(actual_qualities) and actual_qualities[i] != expected:
                    logger.error(f"❌ Quality level {i} is {actual_qualities[i]}, expected {expected}")
                    return False
            
            # Verify memory limits are decreasing
            memory_limits = [level["max_memory_mb"] for level in quality_levels]
            for i in range(1, len(memory_limits)):
                if memory_limits[i] >= memory_limits[i-1]:
                    logger.error(f"❌ Memory limits not decreasing: {memory_limits}")
                    return False
            
            # Verify timeout multipliers are decreasing
            timeout_multipliers = [level["timeout_multiplier"] for level in quality_levels]
            for i in range(1, len(timeout_multipliers)):
                if timeout_multipliers[i] >= timeout_multipliers[i-1]:
                    logger.error(f"❌ Timeout multipliers not decreasing: {timeout_multipliers}")
                    return False
            
            logger.info(f"✅ Progressive fallback configuration valid: {actual_qualities}")
            logger.info(f"✅ Memory limits: {memory_limits}")
            logger.info(f"✅ Timeout multipliers: {timeout_multipliers}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Progressive fallback simulation failed: {e}")
            return False
    
    async def test_memory_constraint_handling(self) -> bool:
        """Test memory constraint handling"""
        try:
            logger.info("Testing memory constraint handling...")
            
            # Get current system memory
            system_memory = psutil.virtual_memory()
            available_memory_mb = system_memory.available / (1024 * 1024)
            
            logger.info(f"System memory - Total: {system_memory.total / (1024**3):.1f}GB, "
                       f"Available: {available_memory_mb:.1f}MB")
            
            # Test quality recommendation based on actual available memory
            recommended_quality = self.downloader.get_quality_recommendation(int(available_memory_mb))
            logger.info(f"✅ Recommended quality for current system: {recommended_quality}")
            
            # Test with constrained memory scenarios
            constrained_scenarios = [
                (512, "worst"),   # Very low memory
                (1536, "360p"),   # Low memory
                (3072, "480p"),   # Medium memory
                (6144, "720p"),   # High memory
            ]
            
            for memory_mb, expected_min_quality in constrained_scenarios:
                recommended = self.downloader.get_quality_recommendation(memory_mb)
                
                # Quality hierarchy: worst < 360p < 480p < 720p
                quality_hierarchy = ["worst", "360p", "480p", "720p"]
                recommended_level = quality_hierarchy.index(recommended)
                expected_level = quality_hierarchy.index(expected_min_quality)
                
                if recommended_level >= expected_level:
                    logger.info(f"✅ {memory_mb}MB -> {recommended} (meets minimum {expected_min_quality})")
                else:
                    logger.error(f"❌ {memory_mb}MB -> {recommended} (below minimum {expected_min_quality})")
                    return False
            
            logger.info("✅ Memory constraint handling tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory constraint handling test failed: {e}")
            return False
    
    async def test_quality_configuration(self) -> bool:
        """Test quality configuration validity"""
        try:
            logger.info("Testing quality configuration...")
            
            quality_levels = self.downloader.quality_levels
            
            # Check that all required fields are present
            required_fields = ["quality", "max_memory_mb", "timeout_multiplier"]
            
            for i, level in enumerate(quality_levels):
                for field in required_fields:
                    if field not in level:
                        logger.error(f"❌ Quality level {i} missing field: {field}")
                        return False
                
                # Validate field types and ranges
                if not isinstance(level["max_memory_mb"], int) or level["max_memory_mb"] <= 0:
                    logger.error(f"❌ Invalid max_memory_mb in level {i}: {level['max_memory_mb']}")
                    return False
                
                if not isinstance(level["timeout_multiplier"], (int, float)) or level["timeout_multiplier"] <= 0:
                    logger.error(f"❌ Invalid timeout_multiplier in level {i}: {level['timeout_multiplier']}")
                    return False
                
                if not isinstance(level["quality"], str) or not level["quality"]:
                    logger.error(f"❌ Invalid quality in level {i}: {level['quality']}")
                    return False
            
            logger.info(f"✅ Quality configuration valid with {len(quality_levels)} levels")
            
            # Log the configuration for reference
            for i, level in enumerate(quality_levels):
                logger.info(f"  Level {i+1}: {level['quality']} - "
                           f"{level['max_memory_mb']}MB - "
                           f"{level['timeout_multiplier']}x timeout")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Quality configuration test failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print test summary"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 QUALITY FALLBACK TEST SUMMARY")
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
        results_file = Path("test_results_quality_fallback.json")
        with open(results_file, "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "errors": error_tests,
                    "success_rate": success_rate
                },
                "test_results": self.test_results,
                "system_info": {
                    "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                    "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
                    "cpu_count": psutil.cpu_count()
                }
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")

async def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Quality Fallback Test Suite")
        print("Usage: python test_quality_fallback.py")
        print("\nThis script tests the progressive quality fallback system.")
        return
    
    test_suite = QualityFallbackTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
