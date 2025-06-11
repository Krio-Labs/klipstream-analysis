#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced GPU Optimization

This script tests all optimization features of the enhanced Parakeet transcriber
including AMP, memory optimization, parallel chunking, and device-specific optimizations.
"""

import asyncio
import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from pydub import AudioSegment

# Import the enhanced transcriber
from raw_pipeline.transcription.handlers.enhanced_parakeet_gpu import (
    EnhancedGPUOptimizedParakeetTranscriber,
    GPUOptimizationConfig,
    EnhancedParakeetGPUHandler
)

class GPUOptimizationTester:
    """Comprehensive tester for GPU optimization features"""
    
    def __init__(self):
        self.test_results = {}
        self.test_audio_files = []
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Setup test environment and create test audio files"""
        print("🔧 Setting up test environment...")
        
        # Create test directories
        test_dir = Path("/tmp/gpu_optimization_tests")
        test_dir.mkdir(exist_ok=True)
        
        # Create test audio files of different sizes
        self._create_test_audio_files(test_dir)
    
    def _create_test_audio_files(self, test_dir: Path):
        """Create test audio files for different scenarios"""
        try:
            # Create short test audio (30 seconds)
            short_audio = AudioSegment.silent(duration=30000)  # 30 seconds
            short_audio = short_audio.set_frame_rate(16000).set_channels(1)
            short_file = test_dir / "short_test.wav"
            short_audio.export(str(short_file), format="wav")
            self.test_audio_files.append(("short", str(short_file), 30))
            
            # Create medium test audio (5 minutes)
            medium_audio = AudioSegment.silent(duration=300000)  # 5 minutes
            medium_audio = medium_audio.set_frame_rate(16000).set_channels(1)
            medium_file = test_dir / "medium_test.wav"
            medium_audio.export(str(medium_file), format="wav")
            self.test_audio_files.append(("medium", str(medium_file), 300))
            
            # Create long test audio (15 minutes)
            long_audio = AudioSegment.silent(duration=900000)  # 15 minutes
            long_audio = long_audio.set_frame_rate(16000).set_channels(1)
            long_file = test_dir / "long_test.wav"
            long_audio.export(str(long_file), format="wav")
            self.test_audio_files.append(("long", str(long_file), 900))
            
            print(f"✅ Created {len(self.test_audio_files)} test audio files")
            
        except Exception as e:
            print(f"⚠️  Failed to create test audio files: {e}")
            print("   Tests will run with existing files if available")
    
    async def run_comprehensive_tests(self):
        """Run comprehensive GPU optimization tests"""
        print("🧪 COMPREHENSIVE GPU OPTIMIZATION TESTS")
        print("=" * 80)
        
        # Test 1: Basic functionality verification
        await self._test_basic_functionality()
        
        # Test 2: AMP performance comparison
        await self._test_amp_performance()
        
        # Test 3: Memory optimization effectiveness
        await self._test_memory_optimization()
        
        # Test 4: Parallel chunking performance
        await self._test_parallel_chunking()
        
        # Test 5: Device-specific optimizations
        await self._test_device_optimizations()
        
        # Test 6: Batch size optimization
        await self._test_batch_size_optimization()
        
        # Test 7: Error handling and fallbacks
        await self._test_error_handling()
        
        # Generate comprehensive report
        self._generate_test_report()
    
    async def _test_basic_functionality(self):
        """Test basic functionality of enhanced transcriber"""
        print("\n1️⃣ BASIC FUNCTIONALITY TEST")
        print("-" * 40)
        
        try:
            config = GPUOptimizationConfig()
            transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)
            
            # Get optimization status
            status = transcriber.get_optimization_status()
            
            print(f"✅ Transcriber initialized successfully")
            print(f"   Device: {status['device']}")
            print(f"   Batch size: {status['batch_size']}")
            print(f"   AMP enabled: {status['amp_enabled']}")
            print(f"   Parallel chunking: {status['parallel_chunking']}")
            
            # Test with shortest audio file
            if self.test_audio_files:
                test_name, test_file, duration = self.test_audio_files[0]
                
                start_time = time.time()
                result = await transcriber.transcribe_long_audio(test_file)
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                self.test_results["basic_functionality"] = {
                    "success": True,
                    "processing_time": processing_time,
                    "audio_duration": duration,
                    "speed_ratio": duration / processing_time if processing_time > 0 else 0,
                    "words_count": len(result.get("words", [])),
                    "segments_count": len(result.get("segments", [])),
                    "performance_metrics": result.get("performance_metrics", {})
                }
                
                print(f"✅ Basic transcription successful")
                print(f"   Processing time: {processing_time:.2f}s")
                print(f"   Speed ratio: {duration/processing_time:.2f}x real-time")
            
            transcriber.cleanup_gpu_resources()
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            self.test_results["basic_functionality"] = {"success": False, "error": str(e)}
    
    async def _test_amp_performance(self):
        """Test AMP performance vs non-AMP"""
        print("\n2️⃣ AMP PERFORMANCE COMPARISON")
        print("-" * 40)
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - skipping AMP test")
            return
        
        try:
            # Test with AMP enabled
            config_amp = GPUOptimizationConfig()
            config_amp.enable_amp = True
            
            # Test with AMP disabled
            config_no_amp = GPUOptimizationConfig()
            config_no_amp.enable_amp = False
            
            amp_results = []
            no_amp_results = []
            
            for test_name, test_file, duration in self.test_audio_files[:2]:  # Test first 2 files
                print(f"   Testing {test_name} audio ({duration}s)...")
                
                # Test with AMP
                transcriber_amp = EnhancedGPUOptimizedParakeetTranscriber(config=config_amp)
                start_time = time.time()
                result_amp = await transcriber_amp.transcribe_long_audio(test_file)
                amp_time = time.time() - start_time
                amp_memory = result_amp.get("performance_metrics", {}).get("peak_memory_gb", 0)
                transcriber_amp.cleanup_gpu_resources()
                
                # Test without AMP
                transcriber_no_amp = EnhancedGPUOptimizedParakeetTranscriber(config=config_no_amp)
                start_time = time.time()
                result_no_amp = await transcriber_no_amp.transcribe_long_audio(test_file)
                no_amp_time = time.time() - start_time
                no_amp_memory = result_no_amp.get("performance_metrics", {}).get("peak_memory_gb", 0)
                transcriber_no_amp.cleanup_gpu_resources()
                
                amp_results.append({"time": amp_time, "memory": amp_memory})
                no_amp_results.append({"time": no_amp_time, "memory": no_amp_memory})
                
                speedup = no_amp_time / amp_time if amp_time > 0 else 0
                memory_savings = (no_amp_memory - amp_memory) / no_amp_memory * 100 if no_amp_memory > 0 else 0
                
                print(f"     AMP: {amp_time:.2f}s, {amp_memory:.2f}GB")
                print(f"     No AMP: {no_amp_time:.2f}s, {no_amp_memory:.2f}GB")
                print(f"     Speedup: {speedup:.2f}x, Memory savings: {memory_savings:.1f}%")
            
            # Calculate averages
            avg_amp_time = np.mean([r["time"] for r in amp_results])
            avg_no_amp_time = np.mean([r["time"] for r in no_amp_results])
            avg_speedup = avg_no_amp_time / avg_amp_time if avg_amp_time > 0 else 0
            
            self.test_results["amp_performance"] = {
                "success": True,
                "average_speedup": avg_speedup,
                "amp_results": amp_results,
                "no_amp_results": no_amp_results
            }
            
            print(f"✅ AMP test completed - Average speedup: {avg_speedup:.2f}x")
            
        except Exception as e:
            print(f"❌ AMP performance test failed: {e}")
            self.test_results["amp_performance"] = {"success": False, "error": str(e)}
    
    async def _test_memory_optimization(self):
        """Test memory optimization effectiveness"""
        print("\n3️⃣ MEMORY OPTIMIZATION TEST")
        print("-" * 40)
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - skipping memory optimization test")
            return
        
        try:
            # Test with memory optimization enabled
            config_opt = GPUOptimizationConfig()
            config_opt.enable_memory_optimization = True
            
            # Test with memory optimization disabled
            config_no_opt = GPUOptimizationConfig()
            config_no_opt.enable_memory_optimization = False
            
            # Use longest audio file for memory stress test
            if len(self.test_audio_files) >= 3:
                test_name, test_file, duration = self.test_audio_files[2]  # Long audio
                
                print(f"   Testing memory optimization with {test_name} audio ({duration}s)...")
                
                # Test with optimization
                transcriber_opt = EnhancedGPUOptimizedParakeetTranscriber(config=config_opt)
                result_opt = await transcriber_opt.transcribe_long_audio(test_file)
                opt_metrics = result_opt.get("performance_metrics", {})
                transcriber_opt.cleanup_gpu_resources()
                
                # Test without optimization
                transcriber_no_opt = EnhancedGPUOptimizedParakeetTranscriber(config=config_no_opt)
                result_no_opt = await transcriber_no_opt.transcribe_long_audio(test_file)
                no_opt_metrics = result_no_opt.get("performance_metrics", {})
                transcriber_no_opt.cleanup_gpu_resources()
                
                opt_memory = opt_metrics.get("peak_memory_gb", 0)
                no_opt_memory = no_opt_metrics.get("peak_memory_gb", 0)
                memory_savings = (no_opt_memory - opt_memory) / no_opt_memory * 100 if no_opt_memory > 0 else 0
                
                self.test_results["memory_optimization"] = {
                    "success": True,
                    "optimized_memory_gb": opt_memory,
                    "unoptimized_memory_gb": no_opt_memory,
                    "memory_savings_percent": memory_savings,
                    "memory_efficiency_optimized": opt_metrics.get("memory_efficiency", 0),
                    "memory_efficiency_unoptimized": no_opt_metrics.get("memory_efficiency", 0)
                }
                
                print(f"✅ Memory optimization test completed")
                print(f"   Optimized: {opt_memory:.2f}GB peak")
                print(f"   Unoptimized: {no_opt_memory:.2f}GB peak")
                print(f"   Memory savings: {memory_savings:.1f}%")
            
        except Exception as e:
            print(f"❌ Memory optimization test failed: {e}")
            self.test_results["memory_optimization"] = {"success": False, "error": str(e)}

    async def _test_parallel_chunking(self):
        """Test parallel chunking performance"""
        print("\n4️⃣ PARALLEL CHUNKING TEST")
        print("-" * 40)

        try:
            # Test with parallel chunking enabled
            config_parallel = GPUOptimizationConfig()
            config_parallel.enable_parallel_chunking = True

            # Test with parallel chunking disabled
            config_sequential = GPUOptimizationConfig()
            config_sequential.enable_parallel_chunking = False

            if len(self.test_audio_files) >= 2:
                test_name, test_file, duration = self.test_audio_files[1]  # Medium audio

                print(f"   Testing parallel chunking with {test_name} audio ({duration}s)...")

                # Test parallel
                transcriber_parallel = EnhancedGPUOptimizedParakeetTranscriber(config=config_parallel)
                start_time = time.time()
                result_parallel = await transcriber_parallel.transcribe_long_audio(test_file)
                parallel_time = time.time() - start_time
                transcriber_parallel.cleanup_gpu_resources()

                # Test sequential
                transcriber_sequential = EnhancedGPUOptimizedParakeetTranscriber(config=config_sequential)
                start_time = time.time()
                result_sequential = await transcriber_sequential.transcribe_long_audio(test_file)
                sequential_time = time.time() - start_time
                transcriber_sequential.cleanup_gpu_resources()

                speedup = sequential_time / parallel_time if parallel_time > 0 else 0

                self.test_results["parallel_chunking"] = {
                    "success": True,
                    "parallel_time": parallel_time,
                    "sequential_time": sequential_time,
                    "speedup": speedup,
                    "parallel_metrics": result_parallel.get("performance_metrics", {}),
                    "sequential_metrics": result_sequential.get("performance_metrics", {})
                }

                print(f"✅ Parallel chunking test completed")
                print(f"   Parallel: {parallel_time:.2f}s")
                print(f"   Sequential: {sequential_time:.2f}s")
                print(f"   Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"❌ Parallel chunking test failed: {e}")
            self.test_results["parallel_chunking"] = {"success": False, "error": str(e)}

    async def _test_device_optimizations(self):
        """Test device-specific optimizations"""
        print("\n5️⃣ DEVICE OPTIMIZATION TEST")
        print("-" * 40)

        try:
            config = GPUOptimizationConfig()
            transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)
            status = transcriber.get_optimization_status()

            device_info = {
                "device": status["device"],
                "device_optimization_enabled": status["device_optimization"],
                "batch_size": status["batch_size"]
            }

            if status["device"] == "cuda" and status["cuda_capabilities"]:
                cuda_caps = status["cuda_capabilities"]
                device_info.update({
                    "compute_capability": cuda_caps["compute_capability"],
                    "memory_gb": cuda_caps["memory_gb"],
                    "tf32_enabled": cuda_caps["tf32_enabled"]
                })

                print(f"✅ CUDA optimizations detected")
                print(f"   Compute capability: {cuda_caps['compute_capability']}")
                print(f"   GPU memory: {cuda_caps['memory_gb']:.1f}GB")
                print(f"   TF32 enabled: {cuda_caps['tf32_enabled']}")

            elif status["device"] == "mps":
                print(f"✅ Apple Silicon MPS optimizations detected")

            else:
                print(f"✅ CPU optimizations detected")
                print(f"   Threads: {torch.get_num_threads()}")

            self.test_results["device_optimizations"] = {
                "success": True,
                **device_info
            }

            transcriber.cleanup_gpu_resources()

        except Exception as e:
            print(f"❌ Device optimization test failed: {e}")
            self.test_results["device_optimizations"] = {"success": False, "error": str(e)}

    async def _test_batch_size_optimization(self):
        """Test different batch sizes for optimal performance"""
        print("\n6️⃣ BATCH SIZE OPTIMIZATION TEST")
        print("-" * 40)

        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - skipping batch size optimization test")
            return

        try:
            batch_sizes = [1, 2, 4, 8, 12, 16]
            batch_results = []

            if self.test_audio_files:
                test_name, test_file, duration = self.test_audio_files[0]  # Short audio for quick testing

                print(f"   Testing batch sizes with {test_name} audio ({duration}s)...")

                for batch_size in batch_sizes:
                    try:
                        config = GPUOptimizationConfig()
                        transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)
                        transcriber.batch_size = batch_size  # Override batch size

                        start_time = time.time()
                        result = await transcriber.transcribe_long_audio(test_file)
                        processing_time = time.time() - start_time

                        metrics = result.get("performance_metrics", {})
                        peak_memory = metrics.get("peak_memory_gb", 0)

                        batch_results.append({
                            "batch_size": batch_size,
                            "processing_time": processing_time,
                            "peak_memory_gb": peak_memory,
                            "success": True
                        })

                        print(f"     Batch {batch_size}: {processing_time:.2f}s, {peak_memory:.2f}GB")

                        transcriber.cleanup_gpu_resources()

                    except Exception as e:
                        batch_results.append({
                            "batch_size": batch_size,
                            "error": str(e),
                            "success": False
                        })
                        print(f"     Batch {batch_size}: Failed - {e}")

                # Find optimal batch size
                successful_results = [r for r in batch_results if r["success"]]
                if successful_results:
                    optimal_batch = min(successful_results, key=lambda x: x["processing_time"])

                    self.test_results["batch_size_optimization"] = {
                        "success": True,
                        "batch_results": batch_results,
                        "optimal_batch_size": optimal_batch["batch_size"],
                        "optimal_time": optimal_batch["processing_time"],
                        "optimal_memory": optimal_batch["peak_memory_gb"]
                    }

                    print(f"✅ Optimal batch size: {optimal_batch['batch_size']} "
                          f"({optimal_batch['processing_time']:.2f}s)")

        except Exception as e:
            print(f"❌ Batch size optimization test failed: {e}")
            self.test_results["batch_size_optimization"] = {"success": False, "error": str(e)}

    async def _test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        print("\n7️⃣ ERROR HANDLING TEST")
        print("-" * 40)

        try:
            config = GPUOptimizationConfig()
            transcriber = EnhancedGPUOptimizedParakeetTranscriber(config=config)

            # Test 1: Invalid audio file
            try:
                await transcriber.transcribe_long_audio("/nonexistent/file.wav")
                error_handling_1 = False
            except Exception:
                error_handling_1 = True
                print("✅ Invalid file error handling: PASS")

            # Test 2: Cleanup after error
            try:
                transcriber.cleanup_gpu_resources()
                error_handling_2 = True
                print("✅ Cleanup after error: PASS")
            except Exception:
                error_handling_2 = False
                print("❌ Cleanup after error: FAIL")

            self.test_results["error_handling"] = {
                "success": True,
                "invalid_file_handling": error_handling_1,
                "cleanup_after_error": error_handling_2
            }

            print("✅ Error handling test completed")

        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.test_results["error_handling"] = {"success": False, "error": str(e)}

    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📊 COMPREHENSIVE TEST REPORT")
        print("=" * 80)

        # Save detailed results to file
        report_file = Path("/tmp/gpu_optimization_test_report.json")

        report_data = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "test_results": self.test_results,
            "summary": self._generate_summary()
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        # Print summary
        summary = report_data["summary"]

        print(f"📈 PERFORMANCE SUMMARY")
        print(f"   Tests passed: {summary['tests_passed']}/{summary['total_tests']}")
        print(f"   Overall success rate: {summary['success_rate']:.1f}%")

        if "basic_functionality" in self.test_results and self.test_results["basic_functionality"]["success"]:
            basic = self.test_results["basic_functionality"]
            print(f"   Basic transcription speed: {basic['speed_ratio']:.2f}x real-time")

        if "amp_performance" in self.test_results and self.test_results["amp_performance"]["success"]:
            amp = self.test_results["amp_performance"]
            print(f"   AMP speedup: {amp['average_speedup']:.2f}x")

        if "memory_optimization" in self.test_results and self.test_results["memory_optimization"]["success"]:
            memory = self.test_results["memory_optimization"]
            print(f"   Memory savings: {memory['memory_savings_percent']:.1f}%")

        if "parallel_chunking" in self.test_results and self.test_results["parallel_chunking"]["success"]:
            parallel = self.test_results["parallel_chunking"]
            print(f"   Parallel chunking speedup: {parallel['speedup']:.2f}x")

        print(f"\n📄 Detailed report saved to: {report_file}")
        print("✅ All tests completed!")

    def _get_system_info(self) -> Dict:
        """Get system information"""
        system_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available()
        }

        if torch.cuda.is_available():
            system_info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "compute_capability": torch.cuda.get_device_capability()
            })

        return system_info

    def _generate_summary(self) -> Dict:
        """Generate test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("success", False))

        return {
            "total_tests": total_tests,
            "tests_passed": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "test_names": list(self.test_results.keys())
        }

async def main():
    """Main test execution"""
    tester = GPUOptimizationTester()
    await tester.run_comprehensive_tests()

if __name__ == "__main__":
    asyncio.run(main())
