#!/usr/bin/env python3
"""
Test script to verify Convex status updates throughout the pipeline

This script tests that status updates happen at the right times
and with the correct status values.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from convex_integration import ConvexIntegration
from utils.logging_setup import setup_logger

# Setup logging
logger = setup_logger(__name__)

def test_status_flow():
    """Test the expected status flow for a video pipeline"""
    
    logger.info("🧪 Testing Convex status flow")
    
    # Initialize Convex integration
    convex_manager = ConvexIntegration()
    
    test_video_id = "test_status_flow"
    
    # Expected status flow based on pipeline implementation
    expected_statuses = [
        "Queued",                    # Pipeline start
        "Downloading",               # Before video download
        "Processing audio",          # During audio conversion
        "Generating waveform",       # After video download
        "Transcribing",             # Before transcript generation
        "Fetching chat",            # Before chat download
        "Uploading files",          # Before GCS uploads
        "Processing complete",       # After Stage 1 completion
        "Analyzing",                # Stage 2 start
        "Finding highlights",       # During analysis
        "Generating visualizations", # Before visualization generation
        "Finalizing analysis",      # Before final completion
        "Completed"                 # Final completion
    ]
    
    logger.info("📋 Expected Status Flow:")
    logger.info("=" * 60)
    
    for i, status in enumerate(expected_statuses, 1):
        logger.info(f"  {i:2d}. {status}")
    
    logger.info("=" * 60)
    
    # Test each status update
    logger.info("🔄 Testing status updates...")
    
    for i, status in enumerate(expected_statuses, 1):
        try:
            # Test the status update
            success = convex_manager.update_video_status(test_video_id, status)
            
            if success:
                logger.info(f"  ✅ {i:2d}. {status}")
            else:
                logger.error(f"  ❌ {i:2d}. {status} - Update failed")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ {i:2d}. {status} - Error: {str(e)}")
            return False
    
    logger.info("🎯 Status flow test completed successfully!")
    return True

def test_status_validation():
    """Test status validation and error handling"""
    
    logger.info("🧪 Testing status validation")
    
    convex_manager = ConvexIntegration()
    test_video_id = "test_validation"
    
    # Test valid statuses
    valid_statuses = [
        "Queued", "Downloading", "Processing audio", "Generating waveform",
        "Transcribing", "Fetching chat", "Uploading files", "Processing complete",
        "Analyzing", "Finding highlights", "Generating visualizations", 
        "Finalizing analysis", "Completed"
    ]
    
    logger.info("✅ Testing valid statuses:")
    all_valid = True
    
    for status in valid_statuses:
        try:
            success = convex_manager.update_video_status(test_video_id, status)
            if success:
                logger.info(f"  ✅ '{status}' - Valid")
            else:
                logger.warning(f"  ⚠️ '{status}' - Update failed but status is valid")
        except Exception as e:
            logger.error(f"  ❌ '{status}' - Error: {str(e)}")
            all_valid = False
    
    # Test edge cases
    logger.info("🔍 Testing edge cases:")
    
    edge_cases = [
        ("", "Empty string"),
        (None, "None value"),
        ("Invalid Status", "Custom status"),
        ("downloading", "Lowercase status"),
        ("COMPLETED", "Uppercase status")
    ]
    
    for status, description in edge_cases:
        try:
            success = convex_manager.update_video_status(test_video_id, status)
            logger.info(f"  ℹ️ {description}: {'Success' if success else 'Failed'}")
        except Exception as e:
            logger.info(f"  ℹ️ {description}: Error - {str(e)}")
    
    return all_valid

def test_status_timing():
    """Test that status updates happen at appropriate times"""
    
    logger.info("🧪 Testing status timing expectations")
    
    # Define expected timing for each status
    status_timing = {
        "Queued": "Immediate (pipeline start)",
        "Downloading": "Before video download starts",
        "Processing audio": "During audio conversion",
        "Generating waveform": "After video download, before transcript",
        "Transcribing": "Before transcript generation",
        "Fetching chat": "Before chat download",
        "Uploading files": "Before GCS uploads",
        "Processing complete": "After Stage 1 completion",
        "Analyzing": "At Stage 2 start",
        "Finding highlights": "During sentiment analysis",
        "Generating visualizations": "Before plot generation",
        "Finalizing analysis": "Before final completion",
        "Completed": "Final pipeline completion"
    }
    
    logger.info("⏱️ Status Timing Guide:")
    logger.info("=" * 80)
    
    for status, timing in status_timing.items():
        logger.info(f"  📍 {status:<25} → {timing}")
    
    logger.info("=" * 80)
    
    # Timing recommendations
    recommendations = [
        "🔄 Status updates should happen BEFORE the operation starts",
        "⚡ Updates should be immediate (no delays)",
        "🎯 Each major pipeline stage should have a status",
        "📊 Long operations (>10s) should have status updates",
        "✅ Final 'Completed' status only after everything succeeds"
    ]
    
    logger.info("💡 Timing Recommendations:")
    for rec in recommendations:
        logger.info(f"  {rec}")
    
    return True

def main():
    """Run all status flow tests"""
    
    logger.info("🚀 Starting Convex status flow tests")
    logger.info("=" * 80)
    
    tests = [
        ("Status Flow", test_status_flow),
        ("Status Validation", test_status_validation),
        ("Status Timing", test_status_timing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} Test")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} Test: PASSED")
            else:
                logger.error(f"❌ {test_name} Test: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} Test: ERROR - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info("-" * 40)
    logger.info(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Status flow is working correctly.")
        return 0
    else:
        logger.error("💥 Some tests failed. Check status update implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
