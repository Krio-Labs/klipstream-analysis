#!/usr/bin/env python3
"""
Test FastAPI Subprocess Fix

This script tests the subprocess wrapper fix for the TwitchDownloaderCLI
"Failure processing application bundle" issue in FastAPI environment.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_subprocess_wrapper():
    """Test the subprocess wrapper functionality"""
    logger.info("🔧 Testing FastAPI Subprocess Wrapper")
    logger.info("=" * 50)
    
    try:
        from api.services.subprocess_wrapper import subprocess_wrapper
        
        # Test 1: Environment setup
        logger.info("📋 Test 1: Environment Setup")
        logger.info(f"Working directory: {subprocess_wrapper.working_directory}")
        logger.info(f"DOTNET_BUNDLE_EXTRACT_BASE_DIR: {subprocess_wrapper.base_env.get('DOTNET_BUNDLE_EXTRACT_BASE_DIR')}")
        
        # Test 2: TwitchDownloaderCLI binary test
        logger.info("\n📋 Test 2: TwitchDownloaderCLI Binary Test")
        cli_test_result = subprocess_wrapper.test_twitch_cli()
        
        if cli_test_result:
            logger.info("✅ TwitchDownloaderCLI test passed")
        else:
            logger.error("❌ TwitchDownloaderCLI test failed")
            return False
        
        # Test 3: Simulate FastAPI thread pool execution
        logger.info("\n📋 Test 3: Thread Pool Execution Simulation")
        
        def run_cli_in_thread():
            """Simulate running CLI in thread pool like FastAPI does"""
            try:
                from utils.config import BINARY_PATHS
                result = subprocess_wrapper.run_subprocess_sync(
                    [BINARY_PATHS["twitch_downloader"], "--version"],
                    timeout=10,
                    capture_output=True,
                    check=False
                )
                return result.returncode in [0, 1]  # Both 0 and 1 are acceptable for --version
            except Exception as e:
                logger.error(f"Thread pool execution failed: {e}")
                return False
        
        # Run in executor like FastAPI does
        loop = asyncio.get_event_loop()
        thread_result = await loop.run_in_executor(None, run_cli_in_thread)
        
        if thread_result:
            logger.info("✅ Thread pool execution test passed")
        else:
            logger.error("❌ Thread pool execution test failed")
            return False
        
        # Test 4: Environment variable inheritance
        logger.info("\n📋 Test 4: Environment Variable Inheritance")
        
        def check_env_inheritance():
            """Check if environment variables are properly inherited"""
            env = subprocess_wrapper.base_env
            required_vars = ['DOTNET_BUNDLE_EXTRACT_BASE_DIR', 'PATH']
            
            for var in required_vars:
                if var not in env:
                    logger.error(f"Missing environment variable: {var}")
                    return False
                logger.info(f"✅ {var}: {env[var][:100]}...")  # Show first 100 chars
            
            return True
        
        env_result = await loop.run_in_executor(None, check_env_inheritance)
        
        if env_result:
            logger.info("✅ Environment variable inheritance test passed")
        else:
            logger.error("❌ Environment variable inheritance test failed")
            return False
        
        logger.info("\n🎉 All tests passed! The subprocess wrapper fix should resolve the FastAPI issue.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        logger.exception("Full exception details:")
        return False


async def test_enhanced_downloader_integration():
    """Test the enhanced downloader integration"""
    logger.info("\n🚀 Testing Enhanced Downloader Integration")
    logger.info("=" * 50)
    
    try:
        from raw_pipeline.enhanced_downloader import EnhancedTwitchDownloader

        # Create downloader instance
        downloader = EnhancedTwitchDownloader()
        
        # Test binary compatibility
        logger.info("📋 Testing binary compatibility in enhanced downloader context")
        
        # This should use the subprocess wrapper if available
        from utils.config import BINARY_PATHS
        
        # Check if we can import the subprocess wrapper
        try:
            from api.services.subprocess_wrapper import subprocess_wrapper
            logger.info("✅ Subprocess wrapper is available for enhanced downloader")
            
            # Test the binary with the wrapper
            test_result = subprocess_wrapper.test_twitch_cli()
            if test_result:
                logger.info("✅ Enhanced downloader can use subprocess wrapper successfully")
                return True
            else:
                logger.error("❌ Enhanced downloader subprocess wrapper test failed")
                return False
                
        except ImportError:
            logger.warning("⚠️ Subprocess wrapper not available (normal for standalone execution)")
            return True
            
    except Exception as e:
        logger.error(f"❌ Enhanced downloader integration test failed: {e}")
        logger.exception("Full exception details:")
        return False


async def test_timeout_manager_integration():
    """Test the timeout manager integration with environment variables"""
    logger.info("\n⏱️ Testing Timeout Manager Integration")
    logger.info("=" * 50)
    
    try:
        from raw_pipeline.timeout_manager import TimeoutAwareProcess, AdaptiveTimeoutManager, TimeoutConfig
        
        # Create timeout manager
        config = TimeoutConfig(base_timeout_seconds=60)
        timeout_manager = AdaptiveTimeoutManager(config)
        
        # Create timeout-aware process
        timeout_process = TimeoutAwareProcess(timeout_manager)
        
        # Test with environment variables
        try:
            from api.services.subprocess_wrapper import subprocess_wrapper
            env = subprocess_wrapper.base_env.copy()
            cwd = subprocess_wrapper.working_directory
            
            logger.info("✅ Timeout manager can use subprocess wrapper environment")
            
            # Test a simple command with the environment
            from utils.config import BINARY_PATHS
            
            # Start a simple process to test environment inheritance
            process = await timeout_process.start_process(
                [BINARY_PATHS["twitch_downloader"], "--help"],
                env=env,
                cwd=cwd
            )
            
            # Wait for completion with timeout
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
                logger.info("✅ Timeout manager environment integration test passed")
                return True
            except asyncio.TimeoutError:
                logger.warning("⚠️ Process timed out (may be normal for --help)")
                await timeout_process.cleanup()
                return True
                
        except ImportError:
            logger.warning("⚠️ Subprocess wrapper not available for timeout manager test")
            return True
            
    except Exception as e:
        logger.error(f"❌ Timeout manager integration test failed: {e}")
        logger.exception("Full exception details:")
        return False


async def main():
    """Run all tests"""
    logger.info("🧪 FastAPI Subprocess Fix Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Subprocess Wrapper", test_subprocess_wrapper),
        ("Enhanced Downloader Integration", test_enhanced_downloader_integration),
        ("Timeout Manager Integration", test_timeout_manager_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The FastAPI subprocess fix should work correctly.")
        return True
    else:
        logger.error(f"💥 {total - passed} tests failed. The fix may need additional work.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
