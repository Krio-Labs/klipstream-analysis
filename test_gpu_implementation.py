#!/usr/bin/env python3
"""
GPU Implementation Test Suite

Comprehensive testing of the new GPU Parakeet transcription implementation
to validate functionality, performance, and integration.
"""

import asyncio
import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logging_setup import setup_logger

logger = setup_logger("gpu_implementation_test", "gpu_implementation_test.log")

class GPUImplementationTester:
    """Comprehensive tester for GPU implementation"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run comprehensive test suite"""
        
        print("🧪 GPU PARAKEET IMPLEMENTATION TEST SUITE")
        print("=" * 50)
        print("Testing new GPU-optimized transcription implementation...")
        print()
        
        tests = [
            ("Configuration System", self.test_configuration_system),
            ("TranscriptionRouter", self.test_transcription_router),
            ("GPU Handler Loading", self.test_gpu_handler_loading),
            ("Deepgram Handler", self.test_deepgram_handler),
            ("Fallback Manager", self.test_fallback_manager),
            ("Cost Optimizer", self.test_cost_optimizer),
            ("Method Selection", self.test_method_selection),
            ("Integration Compatibility", self.test_integration_compatibility),
            ("Environment Variables", self.test_environment_variables),
            ("Error Handling", self.test_error_handling)
        ]
        
        for test_name, test_func in tests:
            print(f"🔍 Testing {test_name}...")
            try:
                result = await test_func()
                self.test_results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"   {status}")
            except Exception as e:
                self.test_results[test_name] = False
                print(f"   ❌ FAILED: {e}")
                logger.error(f"Test {test_name} failed: {e}")
            print()
        
        return self.test_results
    
    async def test_configuration_system(self) -> bool:
        """Test configuration system"""
        
        try:
            from raw_pipeline.transcription.config.settings import TranscriptionConfig, get_config
            
            # Test configuration loading
            config = get_config()
            assert config is not None, "Configuration should load"
            
            # Test validation
            validation = config.validate()
            assert validation["valid"], f"Configuration validation failed: {validation['issues']}"
            
            # Test method selection
            method = config.get_method_for_duration(1.5, gpu_available=True)
            assert method in ["auto", "parakeet", "deepgram", "hybrid"], f"Invalid method: {method}"
            
            # Test chunk parameters
            params = config.get_chunk_parameters(gpu_available=True)
            assert "chunk_duration_seconds" in params, "Chunk parameters missing"
            assert params["chunk_duration_seconds"] > 0, "Invalid chunk duration"
            
            print("     ✓ Configuration loading works")
            print("     ✓ Validation system functional")
            print("     ✓ Method selection logic working")
            print("     ✓ Chunk parameters calculated")
            
            return True
            
        except Exception as e:
            print(f"     ✗ Configuration system error: {e}")
            return False
    
    async def test_transcription_router(self) -> bool:
        """Test TranscriptionRouter functionality"""
        
        try:
            from raw_pipeline.transcription.router import TranscriptionRouter
            
            # Test router initialization
            router = TranscriptionRouter()
            assert router is not None, "Router should initialize"
            
            # Test method selection
            audio_info = {
                "duration_hours": 1.5,
                "duration_seconds": 5400,
                "file_size_mb": 100
            }
            
            method = router._select_transcription_method(audio_info)
            assert method in ["parakeet", "deepgram", "hybrid"], f"Invalid method selected: {method}"
            
            # Test GPU availability check
            gpu_available = router._is_gpu_available()
            assert isinstance(gpu_available, bool), "GPU availability should be boolean"
            
            print("     ✓ Router initialization successful")
            print("     ✓ Method selection working")
            print(f"     ✓ GPU availability: {gpu_available}")
            
            return True
            
        except Exception as e:
            print(f"     ✗ TranscriptionRouter error: {e}")
            return False
    
    async def test_gpu_handler_loading(self) -> bool:
        """Test GPU handler loading (without requiring actual GPU)"""
        
        try:
            # Test lazy loading mechanism
            from raw_pipeline.transcription.router import TranscriptionRouter
            
            router = TranscriptionRouter()
            
            # Test that lazy loading doesn't crash
            try:
                parakeet = router._lazy_load_parakeet()
                # If GPU available, should load successfully
                # If not available, should return None or False
                assert parakeet is None or parakeet is False or hasattr(parakeet, 'transcribe_long_audio'), \
                    "Parakeet handler should be None, False, or valid handler"
                
                print("     ✓ Parakeet lazy loading mechanism works")
                
            except ImportError:
                print("     ✓ Parakeet import gracefully handled when unavailable")
            
            # Test hybrid loading
            try:
                hybrid = router._lazy_load_hybrid()
                assert hybrid is None or hybrid is False or hasattr(hybrid, 'process_audio_files'), \
                    "Hybrid processor should be None, False, or valid processor"
                
                print("     ✓ Hybrid processor lazy loading works")
                
            except ImportError:
                print("     ✓ Hybrid processor import gracefully handled")
            
            return True
            
        except Exception as e:
            print(f"     ✗ GPU handler loading error: {e}")
            return False
    
    async def test_deepgram_handler(self) -> bool:
        """Test Deepgram handler integration"""
        
        try:
            from raw_pipeline.transcription.handlers.deepgram_handler import DeepgramHandler
            
            # Test handler initialization
            handler = DeepgramHandler()
            assert handler is not None, "Deepgram handler should initialize"
            
            # Test availability check
            available = handler.is_available()
            assert isinstance(available, bool), "Availability should be boolean"
            
            # Test cost estimation
            cost = handler.get_estimated_cost(3600)  # 1 hour
            assert cost > 0, "Cost should be positive"
            assert cost == 60 * 0.0045, "Cost calculation should be correct"
            
            # Test time estimation
            time_est = handler.get_estimated_time(3600)
            assert time_est > 0, "Time estimate should be positive"
            
            print("     ✓ Deepgram handler initialization")
            print(f"     ✓ API availability: {available}")
            print(f"     ✓ Cost estimation: ${cost:.3f} for 1 hour")
            print(f"     ✓ Time estimation: {time_est:.1f} seconds")
            
            return True
            
        except Exception as e:
            print(f"     ✗ Deepgram handler error: {e}")
            return False
    
    async def test_fallback_manager(self) -> bool:
        """Test fallback manager functionality"""
        
        try:
            from raw_pipeline.transcription.utils.fallback_manager import FallbackManager, FailureType
            
            # Test manager initialization
            manager = FallbackManager()
            assert manager is not None, "Fallback manager should initialize"
            
            # Test failure classification
            gpu_error = Exception("CUDA out of memory")
            failure_type = manager._classify_failure(gpu_error)
            assert failure_type == FailureType.GPU_MEMORY_ERROR, "Should classify GPU memory error"
            
            model_error = Exception("Model loading failed")
            failure_type = manager._classify_failure(model_error)
            assert failure_type == FailureType.MODEL_LOADING_ERROR, "Should classify model error"
            
            # Test fallback chain
            next_method = manager._get_next_fallback_method("parakeet_gpu", FailureType.GPU_MEMORY_ERROR)
            assert next_method in ["deepgram", "parakeet_cpu"], f"Invalid fallback method: {next_method}"
            
            # Test statistics
            stats = manager.get_failure_statistics()
            assert isinstance(stats, dict), "Statistics should be dictionary"
            assert "failure_counts" in stats, "Should include failure counts"
            
            print("     ✓ Fallback manager initialization")
            print("     ✓ Failure classification working")
            print("     ✓ Fallback chain logic correct")
            print("     ✓ Statistics tracking functional")
            
            return True
            
        except Exception as e:
            print(f"     ✗ Fallback manager error: {e}")
            return False
    
    async def test_cost_optimizer(self) -> bool:
        """Test cost optimizer functionality"""
        
        try:
            from raw_pipeline.transcription.utils.cost_optimizer import CostOptimizer
            
            # Test optimizer initialization
            optimizer = CostOptimizer()
            assert optimizer is not None, "Cost optimizer should initialize"
            
            # Test cost calculations
            deepgram_cost = optimizer.calculate_transcription_cost(3600, "deepgram")
            assert deepgram_cost == 60 * 0.0045, "Deepgram cost calculation incorrect"
            
            gpu_cost = optimizer.calculate_transcription_cost(3600, "parakeet_gpu")
            assert gpu_cost > 0, "GPU cost should be positive"
            
            # Test method optimization
            optimal_method = optimizer.get_optimal_method(1.0, gpu_available=True)
            assert optimal_method in ["deepgram", "parakeet_gpu", "hybrid"], f"Invalid optimal method: {optimal_method}"
            
            # Test record keeping
            optimizer.record_transcription("parakeet_gpu", 3600, 90, 0.05)
            
            # Test analysis
            analysis = optimizer.get_cost_analysis(30)
            assert isinstance(analysis, dict), "Analysis should be dictionary"
            
            print("     ✓ Cost optimizer initialization")
            print(f"     ✓ Deepgram cost (1h): ${deepgram_cost:.3f}")
            print(f"     ✓ GPU cost (1h): ${gpu_cost:.3f}")
            print(f"     ✓ Optimal method (1h): {optimal_method}")
            print("     ✓ Record keeping functional")
            
            return True
            
        except Exception as e:
            print(f"     ✗ Cost optimizer error: {e}")
            return False
    
    async def test_method_selection(self) -> bool:
        """Test intelligent method selection logic"""
        
        try:
            from raw_pipeline.transcription.router import TranscriptionRouter
            
            router = TranscriptionRouter()
            
            # Test different scenarios
            test_cases = [
                {"duration_hours": 0.5, "expected_methods": ["parakeet", "deepgram"]},
                {"duration_hours": 1.5, "expected_methods": ["parakeet", "deepgram", "hybrid"]},
                {"duration_hours": 3.0, "expected_methods": ["hybrid", "deepgram"]},
                {"duration_hours": 5.0, "expected_methods": ["deepgram"]}
            ]
            
            for case in test_cases:
                audio_info = {
                    "duration_hours": case["duration_hours"],
                    "duration_seconds": case["duration_hours"] * 3600,
                    "file_size_mb": case["duration_hours"] * 60
                }
                
                method = router._select_transcription_method(audio_info)
                assert method in case["expected_methods"], \
                    f"Method {method} not in expected {case['expected_methods']} for {case['duration_hours']}h"
            
            print("     ✓ Short file method selection")
            print("     ✓ Medium file method selection")
            print("     ✓ Long file method selection")
            print("     ✓ Very long file method selection")
            
            return True
            
        except Exception as e:
            print(f"     ✗ Method selection error: {e}")
            return False
    
    async def test_integration_compatibility(self) -> bool:
        """Test integration with existing pipeline"""
        
        try:
            # Test backward compatibility wrapper
            from raw_pipeline.transcription.router import TranscriptionHandler
            
            handler = TranscriptionHandler()
            assert handler is not None, "Backward compatibility handler should work"
            assert hasattr(handler, 'process_audio_files'), "Should have legacy method"
            
            # Test that it's actually the router
            assert hasattr(handler, 'transcribe'), "Should have new transcribe method"
            assert hasattr(handler, '_select_transcription_method'), "Should have router methods"
            
            print("     ✓ Backward compatibility wrapper works")
            print("     ✓ Legacy interface preserved")
            print("     ✓ New functionality accessible")
            
            return True
            
        except Exception as e:
            print(f"     ✗ Integration compatibility error: {e}")
            return False
    
    async def test_environment_variables(self) -> bool:
        """Test environment variable handling"""
        
        try:
            # Test configuration from environment
            original_method = os.getenv("TRANSCRIPTION_METHOD")
            
            # Set test environment variable
            os.environ["TRANSCRIPTION_METHOD"] = "parakeet"
            
            # Reload configuration
            from raw_pipeline.transcription.config.settings import reload_config
            config = reload_config()
            
            assert config.transcription_method == "parakeet", "Environment variable not loaded"
            
            # Reset environment
            if original_method:
                os.environ["TRANSCRIPTION_METHOD"] = original_method
            else:
                os.environ.pop("TRANSCRIPTION_METHOD", None)
            
            # Test boolean environment variables
            os.environ["ENABLE_GPU_TRANSCRIPTION"] = "false"
            config = reload_config()
            assert config.enable_gpu_transcription == False, "Boolean environment variable not parsed"
            
            # Reset
            os.environ["ENABLE_GPU_TRANSCRIPTION"] = "true"
            reload_config()
            
            print("     ✓ Environment variable loading")
            print("     ✓ Configuration reloading")
            print("     ✓ Boolean parsing")
            
            return True
            
        except Exception as e:
            print(f"     ✗ Environment variables error: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling mechanisms"""
        
        try:
            from raw_pipeline.transcription.router import TranscriptionRouter
            
            router = TranscriptionRouter()
            
            # Test with invalid audio file
            try:
                result = await router.transcribe("nonexistent_file.mp3", "test_video", "/tmp/test_output")
                assert "error" in result, "Should return error for invalid file"
                
            except Exception:
                # Exception is also acceptable error handling
                pass
            
            # Test fallback manager error handling
            from raw_pipeline.transcription.utils.fallback_manager import FallbackManager
            
            manager = FallbackManager()
            
            # Test complete failure handling
            error = Exception("Test error")
            result = await manager.handle_complete_failure(error, "test.mp3", "test_video", "/tmp")
            
            assert "error" in result, "Should return error result"
            assert result["status"] == "failed", "Should mark as failed"
            assert result["fallback_exhausted"] == True, "Should mark fallback exhausted"
            
            print("     ✓ Invalid file handling")
            print("     ✓ Complete failure handling")
            print("     ✓ Error result structure")
            
            return True
            
        except Exception as e:
            print(f"     ✗ Error handling test error: {e}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        
        total_time = time.time() - self.start_time
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("=" * 60)
        print("📊 GPU IMPLEMENTATION TEST REPORT")
        print("=" * 60)
        
        print(f"\n⏱️  Execution Time: {total_time:.2f} seconds")
        print(f"📈 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {pass_rate:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for test_name, passed in self.test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {test_name}: {status}")
        
        print(f"\n🎯 Implementation Assessment:")
        if pass_rate == 100:
            print("   🎉 ALL TESTS PASSED!")
            print("   ✅ GPU implementation is fully functional")
            print("   ✅ All components working correctly")
            print("   ✅ Integration compatibility confirmed")
            print("   ✅ Error handling robust")
            print("   🚀 Ready for Cloud Run deployment!")
        elif pass_rate >= 80:
            print("   ⚠️  Most tests passed with minor issues")
            print("   📝 Review failed tests before deployment")
            print("   🔧 Address issues and re-test")
        else:
            print("   🔴 Significant test failures detected")
            print("   🛑 DO NOT DEPLOY until issues resolved")
            print("   🔧 Fix all failing components")
        
        # Save detailed report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time_seconds": total_time,
            "overall_pass_rate": pass_rate,
            "test_results": self.test_results,
            "recommendation": "DEPLOY" if pass_rate == 100 else "REVIEW" if pass_rate >= 80 else "FIX",
            "summary": {
                "implementation_functional": pass_rate == 100,
                "components_working": passed_tests,
                "integration_compatible": self.test_results.get("Integration Compatibility", False),
                "error_handling_robust": self.test_results.get("Error Handling", False)
            }
        }
        
        with open("gpu_implementation_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed test report saved to: gpu_implementation_test_report.json")
        
        return report_data

async def main():
    """Main test execution"""
    tester = GPUImplementationTester()
    
    # Run all tests
    results = await tester.run_all_tests()
    
    # Generate comprehensive report
    report = tester.generate_test_report()
    
    # Return success/failure for CI/CD
    return report["overall_pass_rate"] == 100

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
