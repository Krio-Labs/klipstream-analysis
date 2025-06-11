#!/usr/bin/env python3
"""
Proof-of-Concept Integration Test for Enhanced GPU Optimization

This script tests the integration of the enhanced GPU-optimized Parakeet transcriber
with the existing klipstream-analysis pipeline without modifying the main codebase.
"""

import asyncio
import os
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Import enhanced transcriber
from raw_pipeline.transcription.handlers.enhanced_parakeet_gpu import (
    EnhancedGPUOptimizedParakeetTranscriber,
    GPUOptimizationConfig,
    EnhancedParakeetGPUHandler
)

# Import existing components for comparison
try:
    from raw_pipeline.transcription.handlers.parakeet_gpu import GPUOptimizedParakeetTranscriber
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False

class EnhancedIntegrationTester:
    """Integration tester for enhanced GPU optimization"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="enhanced_gpu_test_"))
        print(f"🔧 Test directory: {self.temp_dir}")
    
    async def run_integration_tests(self):
        """Run comprehensive integration tests"""
        print("🔗 ENHANCED GPU OPTIMIZATION INTEGRATION TESTS")
        print("=" * 80)
        
        # Test 1: Enhanced vs Original Performance
        if ORIGINAL_AVAILABLE:
            await self._test_enhanced_vs_original()
        else:
            print("⚠️  Original transcriber not available - skipping comparison")
        
        # Test 2: Handler Integration
        await self._test_handler_integration()
        
        # Test 3: Configuration Flexibility
        await self._test_configuration_flexibility()
        
        # Test 4: Cloud Run Compatibility
        await self._test_cloud_run_compatibility()
        
        # Test 5: Memory Stress Test
        await self._test_memory_stress()
        
        # Test 6: Error Recovery
        await self._test_error_recovery()
        
        # Generate integration report
        self._generate_integration_report()
    
    async def _test_enhanced_vs_original(self):
        """Compare enhanced transcriber with original implementation"""
        print("\n1️⃣ ENHANCED VS ORIGINAL PERFORMANCE")
        print("-" * 50)
        
        try:
            # Create test audio file
            test_audio = self._create_test_audio(duration_minutes=3)
            
            # Test original transcriber
            print("   Testing original transcriber...")
            original_transcriber = GPUOptimizedParakeetTranscriber()
            
            start_time = time.time()
            original_result = await original_transcriber.transcribe_long_audio(test_audio)
            original_time = time.time() - start_time
            original_transcriber.cleanup_gpu_resources()
            
            # Test enhanced transcriber
            print("   Testing enhanced transcriber...")
            config = GPUOptimizationConfig()
            enhanced_transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)
            
            start_time = time.time()
            enhanced_result = await enhanced_transcriber.transcribe_long_audio(test_audio)
            enhanced_time = time.time() - start_time
            enhanced_metrics = enhanced_result.get("performance_metrics", {})
            enhanced_transcriber.cleanup_gpu_resources()
            
            # Compare results
            speedup = original_time / enhanced_time if enhanced_time > 0 else 0
            
            self.test_results["enhanced_vs_original"] = {
                "success": True,
                "original_time": original_time,
                "enhanced_time": enhanced_time,
                "speedup": speedup,
                "original_words": len(original_result.get("words", [])),
                "enhanced_words": len(enhanced_result.get("words", [])),
                "enhanced_metrics": enhanced_metrics
            }
            
            print(f"✅ Performance comparison completed")
            print(f"   Original: {original_time:.2f}s")
            print(f"   Enhanced: {enhanced_time:.2f}s")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Peak memory: {enhanced_metrics.get('peak_memory_gb', 0):.2f}GB")
            
        except Exception as e:
            print(f"❌ Enhanced vs original test failed: {e}")
            self.test_results["enhanced_vs_original"] = {"success": False, "error": str(e)}
    
    async def _test_handler_integration(self):
        """Test enhanced handler integration with existing pipeline structure"""
        print("\n2️⃣ HANDLER INTEGRATION TEST")
        print("-" * 50)
        
        try:
            # Test enhanced handler
            config = GPUOptimizationConfig()
            handler = EnhancedParakeetGPUHandler(config=config)
            
            # Create test audio
            test_audio = self._create_test_audio(duration_minutes=2)
            video_id = "test_video_123"
            
            # Test handler processing
            start_time = time.time()
            result = await handler.process_audio_files(
                video_id=video_id,
                audio_file_path=test_audio,
                output_dir=str(self.temp_dir)
            )
            processing_time = time.time() - start_time
            
            # Verify output files
            transcript_file = Path(result.get("transcript_file", ""))
            words_file = Path(result.get("words_file", ""))
            
            files_exist = transcript_file.exists() and words_file.exists()
            
            # Check file contents
            transcript_valid = False
            words_valid = False
            
            if transcript_file.exists():
                with open(transcript_file, 'r') as f:
                    transcript_data = json.load(f)
                    transcript_valid = "text" in transcript_data and "metadata" in transcript_data
            
            if words_file.exists():
                with open(words_file, 'r') as f:
                    lines = f.readlines()
                    words_valid = len(lines) > 1  # Header + at least one word
            
            handler.cleanup_gpu_resources()
            
            self.test_results["handler_integration"] = {
                "success": True,
                "processing_time": processing_time,
                "files_created": files_exist,
                "transcript_valid": transcript_valid,
                "words_valid": words_valid,
                "output_files": result
            }
            
            print(f"✅ Handler integration test completed")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Files created: {'✅' if files_exist else '❌'}")
            print(f"   Transcript valid: {'✅' if transcript_valid else '❌'}")
            print(f"   Words valid: {'✅' if words_valid else '❌'}")
            
        except Exception as e:
            print(f"❌ Handler integration test failed: {e}")
            self.test_results["handler_integration"] = {"success": False, "error": str(e)}
    
    async def _test_configuration_flexibility(self):
        """Test different configuration options"""
        print("\n3️⃣ CONFIGURATION FLEXIBILITY TEST")
        print("-" * 50)
        
        try:
            test_audio = self._create_test_audio(duration_minutes=1)
            
            # Test different configurations
            configs = [
                ("default", GPUOptimizationConfig()),
                ("amp_disabled", self._create_config(enable_amp=False)),
                ("memory_opt_disabled", self._create_config(enable_memory_optimization=False)),
                ("parallel_disabled", self._create_config(enable_parallel_chunking=False)),
                ("all_disabled", self._create_config(
                    enable_amp=False,
                    enable_memory_optimization=False,
                    enable_parallel_chunking=False
                ))
            ]
            
            config_results = {}
            
            for config_name, config in configs:
                print(f"   Testing {config_name} configuration...")
                
                try:
                    transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)
                    
                    start_time = time.time()
                    result = await transcriber.transcribe_long_audio(test_audio)
                    processing_time = time.time() - start_time
                    
                    metrics = result.get("performance_metrics", {})
                    status = transcriber.get_optimization_status()
                    
                    config_results[config_name] = {
                        "success": True,
                        "processing_time": processing_time,
                        "peak_memory_gb": metrics.get("peak_memory_gb", 0),
                        "amp_enabled": status["amp_enabled"],
                        "parallel_chunking": status["parallel_chunking"],
                        "memory_optimization": status["memory_optimization"]
                    }
                    
                    transcriber.cleanup_gpu_resources()
                    
                    print(f"     ✅ {config_name}: {processing_time:.2f}s")
                    
                except Exception as e:
                    config_results[config_name] = {"success": False, "error": str(e)}
                    print(f"     ❌ {config_name}: {e}")
            
            self.test_results["configuration_flexibility"] = {
                "success": True,
                "config_results": config_results
            }
            
            print(f"✅ Configuration flexibility test completed")
            
        except Exception as e:
            print(f"❌ Configuration flexibility test failed: {e}")
            self.test_results["configuration_flexibility"] = {"success": False, "error": str(e)}
    
    async def _test_cloud_run_compatibility(self):
        """Test Cloud Run environment compatibility"""
        print("\n4️⃣ CLOUD RUN COMPATIBILITY TEST")
        print("-" * 50)
        
        try:
            # Simulate Cloud Run environment variables
            original_env = {}
            cloud_run_env = {
                "ENABLE_AMP": "true",
                "ENABLE_MEMORY_OPTIMIZATION": "true",
                "ENABLE_PARALLEL_CHUNKING": "true",
                "ENABLE_DEVICE_OPTIMIZATION": "true",
                "ENABLE_PERFORMANCE_MONITORING": "true",
                "SAVE_PERFORMANCE_METRICS": "true"
            }
            
            # Set environment variables
            for key, value in cloud_run_env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # Test with Cloud Run configuration
                config = GPUOptimizationConfig()
                transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)
                
                # Verify configuration loaded correctly
                status = transcriber.get_optimization_status()
                
                # Test basic functionality
                test_audio = self._create_test_audio(duration_minutes=1)
                result = await transcriber.transcribe_long_audio(test_audio)
                
                transcriber.cleanup_gpu_resources()
                
                self.test_results["cloud_run_compatibility"] = {
                    "success": True,
                    "config_loaded": True,
                    "amp_enabled": status["amp_enabled"],
                    "memory_optimization": status["memory_optimization"],
                    "parallel_chunking": status["parallel_chunking"],
                    "performance_monitoring": status["performance_monitoring"],
                    "transcription_successful": len(result.get("words", [])) > 0
                }
                
                print(f"✅ Cloud Run compatibility test completed")
                print(f"   AMP enabled: {status['amp_enabled']}")
                print(f"   Memory optimization: {status['memory_optimization']}")
                print(f"   Parallel chunking: {status['parallel_chunking']}")
                
            finally:
                # Restore original environment
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            
        except Exception as e:
            print(f"❌ Cloud Run compatibility test failed: {e}")
            self.test_results["cloud_run_compatibility"] = {"success": False, "error": str(e)}
    
    def _create_config(self, **kwargs) -> GPUOptimizationConfig:
        """Create configuration with specific settings"""
        config = GPUOptimizationConfig()
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config
    
    def _create_test_audio(self, duration_minutes: int = 2) -> str:
        """Create test audio file"""
        try:
            from pydub import AudioSegment
            
            # Create silent audio
            duration_ms = duration_minutes * 60 * 1000
            audio = AudioSegment.silent(duration=duration_ms)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Save to temp file
            test_file = self.temp_dir / f"test_audio_{duration_minutes}min.wav"
            audio.export(str(test_file), format="wav")
            
            return str(test_file)
            
        except Exception as e:
            print(f"Failed to create test audio: {e}")
            raise

    async def _test_memory_stress(self):
        """Test memory handling under stress conditions"""
        print("\n5️⃣ MEMORY STRESS TEST")
        print("-" * 50)

        try:
            # Create longer audio for stress testing
            test_audio = self._create_test_audio(duration_minutes=10)

            config = GPUOptimizationConfig()
            config.enable_memory_optimization = True

            transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)

            # Monitor memory during transcription
            start_time = time.time()
            result = await transcriber.transcribe_long_audio(test_audio)
            processing_time = time.time() - start_time

            metrics = result.get("performance_metrics", {})
            peak_memory = metrics.get("peak_memory_gb", 0)
            memory_efficiency = metrics.get("memory_efficiency", 0)

            transcriber.cleanup_gpu_resources()

            # Check if memory usage is reasonable
            memory_reasonable = peak_memory < 20.0  # Less than 20GB for 10min audio

            self.test_results["memory_stress"] = {
                "success": True,
                "processing_time": processing_time,
                "peak_memory_gb": peak_memory,
                "memory_efficiency": memory_efficiency,
                "memory_reasonable": memory_reasonable,
                "audio_duration_minutes": 10
            }

            print(f"✅ Memory stress test completed")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Peak memory: {peak_memory:.2f}GB")
            print(f"   Memory efficiency: {memory_efficiency:.1f}%")
            print(f"   Memory reasonable: {'✅' if memory_reasonable else '❌'}")

        except Exception as e:
            print(f"❌ Memory stress test failed: {e}")
            self.test_results["memory_stress"] = {"success": False, "error": str(e)}

    async def _test_error_recovery(self):
        """Test error recovery and cleanup"""
        print("\n6️⃣ ERROR RECOVERY TEST")
        print("-" * 50)

        try:
            config = GPUOptimizationConfig()
            transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)

            # Test 1: Invalid file handling
            error_recovery_1 = False
            try:
                await transcriber.transcribe_long_audio("/nonexistent/file.wav")
            except Exception:
                error_recovery_1 = True  # Expected to fail

            # Test 2: Cleanup after error
            cleanup_successful = False
            try:
                transcriber.cleanup_gpu_resources()
                cleanup_successful = True
            except Exception as e:
                print(f"   Cleanup failed: {e}")

            # Test 3: Recovery after error
            recovery_successful = False
            try:
                # Create new transcriber after error
                new_transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)
                test_audio = self._create_test_audio(duration_minutes=1)
                result = await new_transcriber.transcribe_long_audio(test_audio)
                recovery_successful = len(result.get("words", [])) > 0
                new_transcriber.cleanup_gpu_resources()
            except Exception as e:
                print(f"   Recovery failed: {e}")

            self.test_results["error_recovery"] = {
                "success": True,
                "invalid_file_handling": error_recovery_1,
                "cleanup_after_error": cleanup_successful,
                "recovery_after_error": recovery_successful
            }

            print(f"✅ Error recovery test completed")
            print(f"   Invalid file handling: {'✅' if error_recovery_1 else '❌'}")
            print(f"   Cleanup after error: {'✅' if cleanup_successful else '❌'}")
            print(f"   Recovery after error: {'✅' if recovery_successful else '❌'}")

        except Exception as e:
            print(f"❌ Error recovery test failed: {e}")
            self.test_results["error_recovery"] = {"success": False, "error": str(e)}

    def _generate_integration_report(self):
        """Generate comprehensive integration report"""
        print("\n📊 INTEGRATION TEST REPORT")
        print("=" * 80)

        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"📈 INTEGRATION SUMMARY")
        print(f"   Tests passed: {passed_tests}/{total_tests}")
        print(f"   Success rate: {success_rate:.1f}%")

        # Performance highlights
        if "enhanced_vs_original" in self.test_results and self.test_results["enhanced_vs_original"]["success"]:
            comparison = self.test_results["enhanced_vs_original"]
            print(f"   Performance improvement: {comparison['speedup']:.2f}x faster")

        if "memory_stress" in self.test_results and self.test_results["memory_stress"]["success"]:
            memory = self.test_results["memory_stress"]
            print(f"   Memory efficiency: {memory['memory_efficiency']:.1f}%")
            print(f"   Peak memory usage: {memory['peak_memory_gb']:.2f}GB")

        # Configuration flexibility
        if "configuration_flexibility" in self.test_results and self.test_results["configuration_flexibility"]["success"]:
            config_results = self.test_results["configuration_flexibility"]["config_results"]
            working_configs = sum(1 for result in config_results.values() if result.get("success", False))
            print(f"   Configuration flexibility: {working_configs}/{len(config_results)} configs working")

        # Save detailed report
        report_file = self.temp_dir / "integration_test_report.json"

        report_data = {
            "timestamp": time.time(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": success_rate
            },
            "test_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n📄 Detailed report saved to: {report_file}")

        # Print recommendations
        recommendations = report_data["recommendations"]
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        print("✅ Integration testing completed!")

    def _generate_recommendations(self) -> list:
        """Generate recommendations based on test results"""
        recommendations = []

        # Performance recommendations
        if "enhanced_vs_original" in self.test_results and self.test_results["enhanced_vs_original"]["success"]:
            speedup = self.test_results["enhanced_vs_original"]["speedup"]
            if speedup > 1.5:
                recommendations.append(f"Enhanced transcriber shows {speedup:.1f}x speedup - recommend deployment")
            elif speedup > 1.0:
                recommendations.append(f"Enhanced transcriber shows modest {speedup:.1f}x improvement - consider deployment")
            else:
                recommendations.append("Enhanced transcriber performance needs optimization before deployment")

        # Memory recommendations
        if "memory_stress" in self.test_results and self.test_results["memory_stress"]["success"]:
            memory = self.test_results["memory_stress"]
            if memory["memory_reasonable"]:
                recommendations.append("Memory usage is within acceptable limits for Cloud Run deployment")
            else:
                recommendations.append("Memory usage may be too high for Cloud Run - consider optimization")

        # Configuration recommendations
        if "configuration_flexibility" in self.test_results and self.test_results["configuration_flexibility"]["success"]:
            config_results = self.test_results["configuration_flexibility"]["config_results"]
            if "amp_disabled" in config_results and config_results["amp_disabled"]["success"]:
                recommendations.append("AMP fallback working - good for older GPU compatibility")
            if "parallel_disabled" in config_results and config_results["parallel_disabled"]["success"]:
                recommendations.append("Sequential processing fallback working - good for CPU deployment")

        # Error handling recommendations
        if "error_recovery" in self.test_results and self.test_results["error_recovery"]["success"]:
            recovery = self.test_results["error_recovery"]
            if all(recovery[key] for key in ["invalid_file_handling", "cleanup_after_error", "recovery_after_error"]):
                recommendations.append("Excellent error handling - ready for production deployment")
            else:
                recommendations.append("Error handling needs improvement before production deployment")

        return recommendations

async def main():
    """Main integration test execution"""
    tester = EnhancedIntegrationTester()
    try:
        await tester.run_integration_tests()
    finally:
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(tester.temp_dir)
            print(f"🧹 Cleaned up test directory: {tester.temp_dir}")
        except Exception as e:
            print(f"⚠️  Failed to cleanup test directory: {e}")

if __name__ == "__main__":
    asyncio.run(main())
