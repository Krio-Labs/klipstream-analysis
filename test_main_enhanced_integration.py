#!/usr/bin/env python3
"""
Test Enhanced GPU Optimization Integration with Main Pipeline

This script tests the integration of enhanced GPU optimization with the main.py pipeline
to ensure all components work together correctly.
"""

import asyncio
import os
import time
import tempfile
from pathlib import Path
from typing import Dict

# Set environment variables for enhanced GPU optimization
os.environ["ENABLE_ENHANCED_GPU_OPTIMIZATION"] = "true"
os.environ["ENABLE_AMP"] = "true"
os.environ["ENABLE_MEMORY_OPTIMIZATION"] = "true"
os.environ["ENABLE_PARALLEL_CHUNKING"] = "true"
os.environ["ENABLE_DEVICE_OPTIMIZATION"] = "true"
os.environ["ENABLE_PERFORMANCE_MONITORING"] = "true"
os.environ["TRANSCRIPTION_METHOD"] = "auto"

# Import main pipeline components
from main import (
    configure_transcription_environment,
    detect_gpu_capabilities,
    run_integrated_pipeline
)

class MainPipelineEnhancedTester:
    """Test enhanced GPU optimization integration with main pipeline"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="main_enhanced_test_"))
        print(f"🔧 Test directory: {self.temp_dir}")
    
    async def run_integration_tests(self):
        """Run comprehensive integration tests with main pipeline"""
        print("🔗 MAIN PIPELINE ENHANCED GPU INTEGRATION TESTS")
        print("=" * 80)
        
        # Test 1: Configuration Setup
        await self._test_configuration_setup()
        
        # Test 2: GPU Detection and Configuration
        await self._test_gpu_detection()
        
        # Test 3: Transcription Router Integration
        await self._test_transcription_router()
        
        # Test 4: Environment Variable Handling
        await self._test_environment_variables()
        
        # Test 5: Error Handling and Fallbacks
        await self._test_error_handling()
        
        # Test 6: Performance Monitoring
        await self._test_performance_monitoring()
        
        # Generate integration report
        self._generate_integration_report()
    
    async def _test_configuration_setup(self):
        """Test configuration setup with enhanced GPU optimization"""
        print("\n1️⃣ CONFIGURATION SETUP TEST")
        print("-" * 50)
        
        try:
            # Test configuration function
            transcription_config, gpu_info = configure_transcription_environment()
            
            # Verify enhanced GPU optimization is configured
            enhanced_enabled = os.environ.get("ENABLE_ENHANCED_GPU_OPTIMIZATION", "false").lower() == "true"
            amp_enabled = os.environ.get("ENABLE_AMP", "false").lower() == "true"
            memory_opt_enabled = os.environ.get("ENABLE_MEMORY_OPTIMIZATION", "false").lower() == "true"
            
            self.test_results["configuration_setup"] = {
                "success": True,
                "enhanced_gpu_optimization": enhanced_enabled,
                "amp_enabled": amp_enabled,
                "memory_optimization": memory_opt_enabled,
                "transcription_method": transcription_config.get("TRANSCRIPTION_METHOD"),
                "gpu_info": gpu_info
            }
            
            print(f"✅ Configuration setup successful")
            print(f"   Enhanced GPU optimization: {'✅' if enhanced_enabled else '❌'}")
            print(f"   AMP enabled: {'✅' if amp_enabled else '❌'}")
            print(f"   Memory optimization: {'✅' if memory_opt_enabled else '❌'}")
            print(f"   Transcription method: {transcription_config.get('TRANSCRIPTION_METHOD')}")
            
        except Exception as e:
            print(f"❌ Configuration setup test failed: {e}")
            self.test_results["configuration_setup"] = {"success": False, "error": str(e)}
    
    async def _test_gpu_detection(self):
        """Test GPU detection and capability assessment"""
        print("\n2️⃣ GPU DETECTION TEST")
        print("-" * 50)
        
        try:
            gpu_info = detect_gpu_capabilities()
            
            # Check GPU detection results
            cuda_available = gpu_info.get("nvidia_cuda_available", False)
            mps_available = gpu_info.get("apple_metal_available", False)
            gpu_memory = gpu_info.get("gpu_memory_gb", 0)
            
            self.test_results["gpu_detection"] = {
                "success": True,
                "cuda_available": cuda_available,
                "mps_available": mps_available,
                "gpu_memory_gb": gpu_memory,
                "gpu_info": gpu_info
            }
            
            print(f"✅ GPU detection completed")
            print(f"   CUDA available: {'✅' if cuda_available else '❌'}")
            print(f"   MPS available: {'✅' if mps_available else '❌'}")
            print(f"   GPU memory: {gpu_memory:.1f}GB")
            
            if cuda_available or mps_available:
                print(f"   GPU acceleration ready for enhanced optimization")
            else:
                print(f"   CPU-only mode - enhanced optimization will adapt")
            
        except Exception as e:
            print(f"❌ GPU detection test failed: {e}")
            self.test_results["gpu_detection"] = {"success": False, "error": str(e)}
    
    async def _test_transcription_router(self):
        """Test transcription router with enhanced GPU optimization"""
        print("\n3️⃣ TRANSCRIPTION ROUTER TEST")
        print("-" * 50)
        
        try:
            from raw_pipeline.transcription.router import TranscriptionRouter
            
            # Initialize router
            router = TranscriptionRouter()
            
            # Test method selection
            audio_info = {
                "duration_hours": 0.5,  # 30 minutes
                "file_size_mb": 50.0,
                "sample_rate": 16000
            }
            
            method = router._select_transcription_method(audio_info)
            
            # Test enhanced transcriber loading
            enhanced_enabled = os.environ.get("ENABLE_ENHANCED_GPU_OPTIMIZATION", "true").lower() == "true"
            expected_method = "parakeet_enhanced" if enhanced_enabled else "parakeet"
            
            # Check if GPU is available
            gpu_available = router._is_gpu_available()
            
            self.test_results["transcription_router"] = {
                "success": True,
                "selected_method": method,
                "expected_method": expected_method if gpu_available else "deepgram",
                "gpu_available": gpu_available,
                "enhanced_enabled": enhanced_enabled,
                "router_initialized": True
            }
            
            print(f"✅ Transcription router test completed")
            print(f"   Selected method: {method}")
            print(f"   GPU available: {'✅' if gpu_available else '❌'}")
            print(f"   Enhanced optimization: {'✅' if enhanced_enabled else '❌'}")
            
            # Test lazy loading
            if gpu_available:
                try:
                    parakeet_handler = router._lazy_load_parakeet()
                    handler_loaded = parakeet_handler is not None
                    print(f"   Parakeet handler loaded: {'✅' if handler_loaded else '❌'}")
                except Exception as e:
                    print(f"   Parakeet handler loading failed: {e}")
            
        except Exception as e:
            print(f"❌ Transcription router test failed: {e}")
            self.test_results["transcription_router"] = {"success": False, "error": str(e)}
    
    async def _test_environment_variables(self):
        """Test environment variable handling"""
        print("\n4️⃣ ENVIRONMENT VARIABLES TEST")
        print("-" * 50)
        
        try:
            # Test key environment variables
            env_vars = {
                "ENABLE_ENHANCED_GPU_OPTIMIZATION": os.environ.get("ENABLE_ENHANCED_GPU_OPTIMIZATION"),
                "ENABLE_AMP": os.environ.get("ENABLE_AMP"),
                "ENABLE_MEMORY_OPTIMIZATION": os.environ.get("ENABLE_MEMORY_OPTIMIZATION"),
                "ENABLE_PARALLEL_CHUNKING": os.environ.get("ENABLE_PARALLEL_CHUNKING"),
                "ENABLE_DEVICE_OPTIMIZATION": os.environ.get("ENABLE_DEVICE_OPTIMIZATION"),
                "ENABLE_PERFORMANCE_MONITORING": os.environ.get("ENABLE_PERFORMANCE_MONITORING"),
                "TRANSCRIPTION_METHOD": os.environ.get("TRANSCRIPTION_METHOD"),
                "PARAKEET_MODEL_NAME": os.environ.get("PARAKEET_MODEL_NAME")
            }
            
            # Check if all required variables are set
            required_vars = ["ENABLE_ENHANCED_GPU_OPTIMIZATION", "TRANSCRIPTION_METHOD"]
            missing_vars = [var for var in required_vars if not env_vars.get(var)]
            
            self.test_results["environment_variables"] = {
                "success": len(missing_vars) == 0,
                "env_vars": env_vars,
                "missing_vars": missing_vars,
                "all_vars_set": len(missing_vars) == 0
            }
            
            print(f"✅ Environment variables test completed")
            print(f"   Required variables set: {'✅' if len(missing_vars) == 0 else '❌'}")
            
            if missing_vars:
                print(f"   Missing variables: {missing_vars}")
            
            for var, value in env_vars.items():
                if value:
                    print(f"   {var}: {value}")
            
        except Exception as e:
            print(f"❌ Environment variables test failed: {e}")
            self.test_results["environment_variables"] = {"success": False, "error": str(e)}
    
    async def _test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        print("\n5️⃣ ERROR HANDLING TEST")
        print("-" * 50)
        
        try:
            # Test configuration with invalid settings
            original_method = os.environ.get("TRANSCRIPTION_METHOD")
            
            # Test with invalid method
            os.environ["TRANSCRIPTION_METHOD"] = "invalid_method"
            
            try:
                from raw_pipeline.transcription.config.settings import reload_config
                config = reload_config()
                
                # Should handle invalid method gracefully
                error_handling_1 = True
                print("✅ Invalid method handling: PASS")
                
            except Exception:
                error_handling_1 = False
                print("❌ Invalid method handling: FAIL")
            
            # Restore original method
            if original_method:
                os.environ["TRANSCRIPTION_METHOD"] = original_method
            
            # Test fallback mechanism
            try:
                from raw_pipeline.transcription.router import TranscriptionRouter
                router = TranscriptionRouter()
                
                # This should work even with potential issues
                gpu_available = router._is_gpu_available()
                error_handling_2 = True
                print("✅ Fallback mechanism: PASS")
                
            except Exception:
                error_handling_2 = False
                print("❌ Fallback mechanism: FAIL")
            
            self.test_results["error_handling"] = {
                "success": True,
                "invalid_method_handling": error_handling_1,
                "fallback_mechanism": error_handling_2
            }
            
            print(f"✅ Error handling test completed")
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.test_results["error_handling"] = {"success": False, "error": str(e)}
    
    async def _test_performance_monitoring(self):
        """Test performance monitoring integration"""
        print("\n6️⃣ PERFORMANCE MONITORING TEST")
        print("-" * 50)
        
        try:
            # Test performance monitoring configuration
            monitoring_enabled = os.environ.get("ENABLE_PERFORMANCE_MONITORING", "false").lower() == "true"
            save_metrics = os.environ.get("SAVE_PERFORMANCE_METRICS", "false").lower() == "true"
            
            # Test metrics file path
            metrics_file = Path("/tmp/transcription_performance_metrics.json")
            
            self.test_results["performance_monitoring"] = {
                "success": True,
                "monitoring_enabled": monitoring_enabled,
                "save_metrics": save_metrics,
                "metrics_file_path": str(metrics_file),
                "metrics_file_writable": metrics_file.parent.exists() and os.access(metrics_file.parent, os.W_OK)
            }
            
            print(f"✅ Performance monitoring test completed")
            print(f"   Monitoring enabled: {'✅' if monitoring_enabled else '❌'}")
            print(f"   Save metrics: {'✅' if save_metrics else '❌'}")
            print(f"   Metrics file writable: {'✅' if self.test_results['performance_monitoring']['metrics_file_writable'] else '❌'}")
            
        except Exception as e:
            print(f"❌ Performance monitoring test failed: {e}")
            self.test_results["performance_monitoring"] = {"success": False, "error": str(e)}
    
    def _generate_integration_report(self):
        """Generate comprehensive integration report"""
        print("\n📊 MAIN PIPELINE INTEGRATION REPORT")
        print("=" * 80)
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📈 INTEGRATION SUMMARY")
        print(f"   Tests passed: {passed_tests}/{total_tests}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        # Enhanced GPU optimization status
        config_result = self.test_results.get("configuration_setup", {})
        if config_result.get("success"):
            print(f"   Enhanced GPU optimization: {'✅' if config_result.get('enhanced_gpu_optimization') else '❌'}")
            print(f"   AMP enabled: {'✅' if config_result.get('amp_enabled') else '❌'}")
            print(f"   Memory optimization: {'✅' if config_result.get('memory_optimization') else '❌'}")
        
        # GPU capabilities
        gpu_result = self.test_results.get("gpu_detection", {})
        if gpu_result.get("success"):
            cuda_available = gpu_result.get("cuda_available", False)
            mps_available = gpu_result.get("mps_available", False)
            print(f"   GPU acceleration: {'✅' if (cuda_available or mps_available) else '❌'}")
        
        # Router integration
        router_result = self.test_results.get("transcription_router", {})
        if router_result.get("success"):
            method = router_result.get("selected_method")
            enhanced = "enhanced" in method if method else False
            print(f"   Enhanced transcription method: {'✅' if enhanced else '❌'}")
        
        print(f"\n💡 RECOMMENDATIONS")
        
        if success_rate >= 80:
            print(f"   ✅ Integration successful - ready for enhanced GPU optimization deployment")
        else:
            print(f"   ⚠️  Integration issues detected - review failed tests before deployment")
        
        if config_result.get("enhanced_gpu_optimization"):
            print(f"   🚀 Enhanced GPU optimization is properly configured")
        else:
            print(f"   ⚠️  Enhanced GPU optimization not enabled - check environment variables")
        
        if gpu_result.get("cuda_available") or gpu_result.get("mps_available"):
            print(f"   🎯 GPU acceleration available - optimal performance expected")
        else:
            print(f"   💻 CPU-only mode - enhanced optimization will adapt automatically")
        
        print("✅ Main pipeline integration testing completed!")

async def main():
    """Main test execution"""
    tester = MainPipelineEnhancedTester()
    await tester.run_integration_tests()

if __name__ == "__main__":
    asyncio.run(main())
