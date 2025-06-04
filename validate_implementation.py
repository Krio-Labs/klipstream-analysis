#!/usr/bin/env python3
"""
Implementation Validation Script

Quick validation of the incident remediation implementation.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all new modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from raw_pipeline.enhanced_downloader import EnhancedTwitchDownloader
        print("✅ Enhanced downloader import successful")
    except Exception as e:
        print(f"❌ Enhanced downloader import failed: {e}")
        return False
    
    try:
        from raw_pipeline.timeout_manager import AdaptiveTimeoutManager, TimeoutConfig
        print("✅ Timeout manager import successful")
    except Exception as e:
        print(f"❌ Timeout manager import failed: {e}")
        return False
    
    try:
        from api.routes.analysis import router
        print("✅ API routes import successful")
    except Exception as e:
        print(f"❌ API routes import failed: {e}")
        return False
    
    return True

def test_enhanced_downloader_creation():
    """Test creating enhanced downloader instance"""
    print("\n🔍 Testing enhanced downloader creation...")
    
    try:
        from raw_pipeline.enhanced_downloader import EnhancedTwitchDownloader
        downloader = EnhancedTwitchDownloader()
        print("✅ Enhanced downloader instance created successfully")
        
        # Test methods exist
        assert hasattr(downloader, 'download_video_with_monitoring')
        assert hasattr(downloader, 'get_active_downloads')
        assert hasattr(downloader, 'get_download_status')
        print("✅ Required methods exist")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced downloader creation failed: {e}")
        return False

def test_timeout_manager_creation():
    """Test creating timeout manager instance"""
    print("\n🔍 Testing timeout manager creation...")
    
    try:
        from raw_pipeline.timeout_manager import AdaptiveTimeoutManager, TimeoutConfig
        
        config = TimeoutConfig(
            base_timeout_seconds=1800,
            max_timeout_seconds=3600,
            progress_stall_timeout=300,
            adaptive_scaling=True
        )
        print("✅ Timeout config created successfully")
        
        manager = AdaptiveTimeoutManager(config)
        print("✅ Timeout manager instance created successfully")
        
        # Test methods exist
        assert hasattr(manager, 'update_progress')
        assert hasattr(manager, 'check_timeouts')
        assert hasattr(manager, 'get_timeout_status')
        print("✅ Required methods exist")
        
        return True
    except Exception as e:
        print(f"❌ Timeout manager creation failed: {e}")
        return False

def test_error_analyzer():
    """Test error analyzer functionality"""
    print("\n🔍 Testing error analyzer...")
    
    try:
        from raw_pipeline.enhanced_downloader import ErrorAnalyzer, ProcessContext, FailureType
        from datetime import datetime, timezone
        
        analyzer = ErrorAnalyzer()
        print("✅ Error analyzer created successfully")
        
        # Create test context
        context = ProcessContext(
            job_id="test_job",
            video_id="test_video",
            start_time=datetime.now(timezone.utc),
            timeout_seconds=1800,
            max_memory_mb=8192
        )
        
        # Test error classification (this is a static method)
        error_type, analysis = ErrorAnalyzer._classify_error(
            RuntimeError("Test error"),
            "test stdout",
            "test stderr",
            1,
            context
        )
        
        assert error_type == FailureType.UNKNOWN
        assert 'user_message' in analysis
        assert 'technical_details' in analysis
        print("✅ Error classification working")
        
        return True
    except Exception as e:
        print(f"❌ Error analyzer test failed: {e}")
        return False

def test_api_routes():
    """Test API routes configuration"""
    print("\n🔍 Testing API routes...")
    
    try:
        from api.routes.analysis import router
        
        # Check that routes are registered
        routes = [route.path for route in router.routes]
        print(f"✅ API routes found: {routes}")
        
        # Check for both analysis and analyze endpoints
        has_analysis = any('/analysis' in route for route in routes)
        has_analyze = any('/analyze' in route for route in routes)
        
        if has_analysis and has_analyze:
            print("✅ Both /analysis and /analyze endpoints available")
            return True
        else:
            print(f"❌ Missing endpoints - analysis: {has_analysis}, analyze: {has_analyze}")
            return False
            
    except Exception as e:
        print(f"❌ API routes test failed: {e}")
        return False

def test_processor_integration():
    """Test processor integration"""
    print("\n🔍 Testing processor integration...")

    try:
        from raw_pipeline.processor import process_raw_files

        print("✅ Raw pipeline process_raw_files function imported successfully")

        # Check that enhanced downloader is imported
        import raw_pipeline.processor as processor_module
        assert hasattr(processor_module, 'EnhancedTwitchDownloader')
        print("✅ Enhanced downloader available in processor")

        # Check that the function is callable
        assert callable(process_raw_files)
        print("✅ process_raw_files is callable")

        return True
    except Exception as e:
        print(f"❌ Processor integration test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 KlipStream Incident Remediation - Implementation Validation")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Enhanced Downloader Creation", test_enhanced_downloader_creation),
        ("Timeout Manager Creation", test_timeout_manager_creation),
        ("Error Analyzer", test_error_analyzer),
        ("API Routes", test_api_routes),
        ("Processor Integration", test_processor_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🔍 Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ ERROR: {test_name} - {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = len([r for r in results if r[1]])
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("\n🎉 All validation tests passed! Implementation is ready.")
        return 0
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
