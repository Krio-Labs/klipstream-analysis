#!/usr/bin/env python3
"""
GPU vs CPU Performance Comparison for Cloud Run

This script compares the performance of Parakeet transcription between:
1. Local Mac (current baseline)
2. Simulated Cloud Run CPU-only
3. Simulated Cloud Run with GPU

Usage:
    python test_gpu_vs_cpu_performance.py <audio_file_path>
"""

import asyncio
import argparse
import time
import psutil
import os
import sys
import json
from pathlib import Path
from typing import Dict
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from raw_pipeline.transcriber_parakeet import ParakeetTranscriptionHandler
from raw_pipeline.transcriber_parakeet_gpu import GPUOptimizedParakeetTranscriber
from utils.logging_setup import setup_logger

logger = setup_logger("gpu_cpu_comparison", "gpu_cpu_comparison.log")

class PerformanceComparison:
    """Class for comparing GPU vs CPU performance"""
    
    def __init__(self, audio_file_path: str):
        self.audio_file_path = audio_file_path
        self.audio_duration = 0
        self.file_size_mb = 0
        self._get_audio_info()
    
    def _get_audio_info(self):
        """Get audio file information"""
        try:
            from pydub import AudioSegment
            
            self.file_size_mb = os.path.getsize(self.audio_file_path) / (1024 * 1024)
            
            audio = AudioSegment.from_file(self.audio_file_path)
            self.audio_duration = len(audio) / 1000.0
            
            print(f"📁 Audio file: {self.audio_file_path}")
            print(f"📊 File size: {self.file_size_mb:.1f} MB")
            print(f"⏱️  Duration: {self.audio_duration/3600:.2f} hours ({self.audio_duration:.1f} seconds)")
            
        except Exception as e:
            print(f"❌ Error analyzing audio: {e}")
            sys.exit(1)
    
    def _get_system_info(self):
        """Get current system information"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "available_memory_gb": psutil.virtual_memory().available / (1024 * 1024 * 1024),
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "None",
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        }
    
    async def test_current_implementation(self) -> Dict:
        """Test current Parakeet implementation (baseline)"""
        print(f"\n🔵 Testing Current Implementation (Baseline)...")
        
        start_time = time.time()
        initial_system = self._get_system_info()
        
        try:
            transcriber = ParakeetTranscriptionHandler()
            
            model_load_time = time.time() - start_time
            transcription_start = time.time()
            
            result = await transcriber.process_audio_files(
                video_id="performance_test_current",
                audio_file_path=self.audio_file_path,
                output_dir="output/performance_test/current"
            )
            
            transcription_time = time.time() - transcription_start
            total_time = time.time() - start_time
            final_system = self._get_system_info()
            
            if result:
                # Count words and paragraphs
                import pandas as pd
                words_df = pd.read_csv(result['words_file'])
                paragraphs_df = pd.read_csv(result['paragraphs_file'])
                
                return {
                    "status": "success",
                    "implementation": "current",
                    "model_load_time": model_load_time,
                    "transcription_time": transcription_time,
                    "total_time": total_time,
                    "processing_speed_ratio": self.audio_duration / transcription_time,
                    "words_transcribed": len(words_df),
                    "paragraphs_created": len(paragraphs_df),
                    "memory_increase_mb": final_system["memory_mb"] - initial_system["memory_mb"],
                    "system_info": initial_system
                }
            else:
                return {"status": "failed", "implementation": "current", "total_time": total_time}
                
        except Exception as e:
            return {"status": "error", "implementation": "current", "error": str(e), "total_time": time.time() - start_time}
    
    async def test_gpu_optimized(self) -> Dict:
        """Test GPU-optimized implementation"""
        print(f"\n🟢 Testing GPU-Optimized Implementation...")
        
        start_time = time.time()
        initial_system = self._get_system_info()
        
        try:
            transcriber = GPUOptimizedParakeetTranscriber()
            
            model_load_time = time.time() - start_time
            transcription_start = time.time()
            
            result = await transcriber.transcribe_long_audio(self.audio_file_path)
            
            transcription_time = time.time() - transcription_start
            total_time = time.time() - start_time
            final_system = self._get_system_info()
            
            if result and result["text"]:
                return {
                    "status": "success",
                    "implementation": "gpu_optimized",
                    "model_load_time": model_load_time,
                    "transcription_time": transcription_time,
                    "total_time": total_time,
                    "processing_speed_ratio": self.audio_duration / transcription_time,
                    "words_transcribed": len(result["words"]),
                    "segments_created": len(result["segments"]),
                    "memory_increase_mb": final_system["memory_mb"] - initial_system["memory_mb"],
                    "system_info": initial_system,
                    "batch_size": transcriber.batch_size
                }
            else:
                return {"status": "failed", "implementation": "gpu_optimized", "total_time": total_time}
                
        except Exception as e:
            return {"status": "error", "implementation": "gpu_optimized", "error": str(e), "total_time": time.time() - start_time}
    
    def simulate_cloud_run_performance(self, local_results: Dict) -> Dict:
        """Simulate Cloud Run performance based on local results"""
        
        # Performance factors for Cloud Run
        cpu_slowdown_factor = 2.5  # x86 vs ARM, no GPU acceleration
        gpu_speedup_factor = 2.0   # GPU vs CPU processing
        cold_start_penalty = 120   # 2 minutes for model download + initialization
        
        simulations = {}
        
        if local_results["status"] == "success":
            base_transcription_time = local_results["transcription_time"]
            base_model_load_time = local_results["model_load_time"]
            
            # Cloud Run CPU-only simulation
            cpu_transcription_time = base_transcription_time * cpu_slowdown_factor
            cpu_model_load_time = base_model_load_time * 2  # Slower model loading
            cpu_total_time = cpu_transcription_time + cpu_model_load_time + cold_start_penalty
            
            simulations["cloud_run_cpu"] = {
                "implementation": "cloud_run_cpu_simulation",
                "model_load_time": cpu_model_load_time,
                "transcription_time": cpu_transcription_time,
                "total_time": cpu_total_time,
                "cold_start_penalty": cold_start_penalty,
                "processing_speed_ratio": self.audio_duration / cpu_transcription_time,
                "timeout_risk": "high" if cpu_total_time > 3000 else "medium" if cpu_total_time > 1800 else "low"
            }
            
            # Cloud Run GPU simulation
            gpu_transcription_time = base_transcription_time / gpu_speedup_factor
            gpu_model_load_time = base_model_load_time * 1.2  # Slightly slower due to CUDA setup
            gpu_total_time = gpu_transcription_time + gpu_model_load_time + (cold_start_penalty * 0.8)  # Faster cold start with GPU
            
            simulations["cloud_run_gpu"] = {
                "implementation": "cloud_run_gpu_simulation",
                "model_load_time": gpu_model_load_time,
                "transcription_time": gpu_transcription_time,
                "total_time": gpu_total_time,
                "cold_start_penalty": cold_start_penalty * 0.8,
                "processing_speed_ratio": self.audio_duration / gpu_transcription_time,
                "timeout_risk": "low" if gpu_total_time < 1800 else "medium"
            }
        
        return simulations
    
    def generate_report(self, results: Dict):
        """Generate comprehensive performance report"""
        
        print(f"\n" + "="*80)
        print(f"🎯 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print(f"="*80)
        
        print(f"\n📊 Audio File Information:")
        print(f"  Duration: {self.audio_duration/3600:.2f} hours ({self.audio_duration:.1f} seconds)")
        print(f"  File Size: {self.file_size_mb:.1f} MB")
        
        # Performance comparison table
        print(f"\n⚡ Performance Comparison:")
        print(f"{'Implementation':<25} {'Time (min)':<12} {'Speed Ratio':<12} {'Status':<10}")
        print(f"-" * 65)
        
        for impl_name, data in results.items():
            if isinstance(data, dict) and "total_time" in data:
                time_min = data["total_time"] / 60
                speed_ratio = data.get("processing_speed_ratio", 0)
                status = data.get("status", "simulated")
                print(f"{impl_name:<25} {time_min:<12.1f} {speed_ratio:<12.1f} {status:<10}")
        
        # 3-hour projections
        print(f"\n📈 3-Hour Audio Projections:")
        print(f"{'Implementation':<25} {'Est. Time':<15} {'Timeout Risk':<15}")
        print(f"-" * 55)
        
        for impl_name, data in results.items():
            if isinstance(data, dict) and "processing_speed_ratio" in data:
                if data["processing_speed_ratio"] > 0:
                    three_hour_time = (3 * 3600) / data["processing_speed_ratio"] / 60  # minutes
                    timeout_risk = data.get("timeout_risk", "unknown")
                    print(f"{impl_name:<25} {three_hour_time:<15.1f} {timeout_risk:<15}")
        
        # Cost analysis
        print(f"\n💰 Cost Analysis (3-hour audio):")
        costs = {
            "deepgram_api": 3 * 60 * 0.0045,  # $0.0045 per minute
            "cloud_run_cpu": 0.45,  # ~45 minutes * $0.01/minute
            "cloud_run_gpu": 0.06   # ~6 minutes * $0.01/minute
        }
        
        for service, cost in costs.items():
            print(f"  {service:<20}: ${cost:<8.2f}")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        
        if "cloud_run_gpu" in results:
            gpu_data = results["cloud_run_gpu"]
            if gpu_data.get("timeout_risk") == "low":
                print(f"  ✅ GPU Cloud Run: Recommended for all audio lengths")
                print(f"     - Fastest processing (~{gpu_data['total_time']/60:.1f} min for this file)")
                print(f"     - Low timeout risk")
                print(f"     - Cost effective")
            else:
                print(f"  ⚠️  GPU Cloud Run: Use with caution for very long files")
        
        if "cloud_run_cpu" in results:
            cpu_data = results["cloud_run_cpu"]
            if cpu_data.get("timeout_risk") == "high":
                print(f"  🔴 CPU Cloud Run: Not recommended for files > 2 hours")
                print(f"     - High timeout risk ({cpu_data['total_time']/60:.1f} min estimated)")
            else:
                print(f"  🟡 CPU Cloud Run: Acceptable for shorter files")
        
        print(f"  💡 Hybrid approach: Use GPU for long files, CPU for short files")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare GPU vs CPU performance for Parakeet transcription")
    parser.add_argument("audio_file", help="Path to audio file for testing")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU testing")
    parser.add_argument("--output", default="performance_comparison_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Initialize comparison
    comparison = PerformanceComparison(args.audio_file)
    results = {}
    
    # Test current implementation
    current_results = await comparison.test_current_implementation()
    results["current_local"] = current_results
    
    # Test GPU-optimized implementation (if available and not skipped)
    if not args.skip_gpu and torch.cuda.is_available():
        gpu_results = await comparison.test_gpu_optimized()
        results["gpu_optimized_local"] = gpu_results
    elif args.skip_gpu:
        print("🔵 Skipping GPU tests (--skip-gpu flag)")
    else:
        print("🔵 No GPU available for testing")
    
    # Simulate Cloud Run performance
    if current_results["status"] == "success":
        cloud_simulations = comparison.simulate_cloud_run_performance(current_results)
        results.update(cloud_simulations)
    
    # Generate report
    comparison.generate_report(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
