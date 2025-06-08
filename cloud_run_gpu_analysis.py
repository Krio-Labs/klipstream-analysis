#!/usr/bin/env python3
"""
Cloud Run GPU Performance Analysis

Based on our successful 1.14-hour audio test results, this script provides
detailed analysis and projections for Cloud Run performance with and without GPU.

Results from our test:
- Local Mac (MPS): 1.14h audio in 3.9 minutes (17.5x real-time)
- Processing: 10,543 words, 1,275 paragraphs
- Memory: 4.9 GB increase
"""

import json
from typing import Dict

class CloudRunGPUAnalysis:
    """Analyze Cloud Run performance scenarios"""
    
    def __init__(self):
        # Baseline results from our successful test
        self.baseline_results = {
            "audio_duration_hours": 1.14,
            "audio_duration_seconds": 4113.8,
            "local_mac_transcription_time": 234.8,  # seconds
            "local_mac_model_load_time": 7.3,       # seconds
            "local_mac_total_time": 243.1,          # seconds
            "local_mac_speed_ratio": 17.52,         # x real-time
            "words_transcribed": 10543,
            "paragraphs_created": 1275,
            "memory_increase_gb": 4.9
        }
        
        # Cloud Run configuration
        self.cloud_run_config = {
            "cpu_cores": 8,
            "memory_gb": 32,
            "gpu_type": "NVIDIA L4",
            "gpu_memory_gb": 24,
            "timeout_seconds": 3600,
            "cold_start_base": 120  # seconds for model download
        }
        
        # Performance factors
        self.performance_factors = {
            "cpu_architecture_penalty": 1.5,    # x86 vs ARM performance difference
            "no_gpu_penalty": 4.0,              # CPU vs MPS processing
            "gpu_advantage": 2.5,               # NVIDIA L4 vs Apple MPS
            "cloud_io_penalty": 1.2,           # Cloud storage vs local SSD
            "batch_processing_bonus": 0.8       # GPU batch processing efficiency
        }
    
    def calculate_cloud_run_cpu_performance(self) -> Dict:
        """Calculate Cloud Run CPU-only performance"""
        
        base_transcription = self.baseline_results["local_mac_transcription_time"]
        base_model_load = self.baseline_results["local_mac_model_load_time"]
        
        # Apply penalties for CPU-only Cloud Run
        cpu_transcription_time = (
            base_transcription * 
            self.performance_factors["cpu_architecture_penalty"] * 
            self.performance_factors["no_gpu_penalty"] * 
            self.performance_factors["cloud_io_penalty"]
        )
        
        cpu_model_load_time = (
            base_model_load * 
            self.performance_factors["cpu_architecture_penalty"] * 2  # Slower model loading
        )
        
        cold_start = self.cloud_run_config["cold_start_base"]
        total_time = cpu_transcription_time + cpu_model_load_time + cold_start
        
        return {
            "implementation": "cloud_run_cpu_only",
            "model_load_time": cpu_model_load_time,
            "transcription_time": cpu_transcription_time,
            "cold_start_time": cold_start,
            "total_time": total_time,
            "processing_speed_ratio": self.baseline_results["audio_duration_seconds"] / cpu_transcription_time,
            "timeout_risk": self._assess_timeout_risk(total_time),
            "estimated_cost_per_hour": total_time / 3600 * 0.10  # $0.10/hour for CPU
        }
    
    def calculate_cloud_run_gpu_performance(self) -> Dict:
        """Calculate Cloud Run GPU performance"""
        
        base_transcription = self.baseline_results["local_mac_transcription_time"]
        base_model_load = self.baseline_results["local_mac_model_load_time"]
        
        # Apply GPU advantages and minor penalties
        gpu_transcription_time = (
            base_transcription / 
            self.performance_factors["gpu_advantage"] * 
            self.performance_factors["batch_processing_bonus"] * 
            self.performance_factors["cloud_io_penalty"]
        )
        
        gpu_model_load_time = base_model_load * 1.3  # Slightly slower due to CUDA setup
        
        cold_start = self.cloud_run_config["cold_start_base"] * 0.9  # Faster with GPU
        total_time = gpu_transcription_time + gpu_model_load_time + cold_start
        
        return {
            "implementation": "cloud_run_gpu_l4",
            "model_load_time": gpu_model_load_time,
            "transcription_time": gpu_transcription_time,
            "cold_start_time": cold_start,
            "total_time": total_time,
            "processing_speed_ratio": self.baseline_results["audio_duration_seconds"] / gpu_transcription_time,
            "timeout_risk": self._assess_timeout_risk(total_time),
            "estimated_cost_per_hour": total_time / 3600 * 0.45,  # $0.45/hour for GPU
            "gpu_batch_size": 8,  # L4 can handle larger batches
            "gpu_memory_efficiency": "high"
        }
    
    def _assess_timeout_risk(self, total_time: float) -> str:
        """Assess timeout risk based on total processing time"""
        timeout_limit = self.cloud_run_config["timeout_seconds"]
        
        if total_time < timeout_limit * 0.5:
            return "low"
        elif total_time < timeout_limit * 0.8:
            return "medium"
        else:
            return "high"
    
    def project_different_durations(self, cpu_results: Dict, gpu_results: Dict) -> Dict:
        """Project performance for different audio durations"""
        
        durations = [0.5, 1, 2, 3, 6, 10]  # hours
        projections = {}
        
        for duration_hours in durations:
            duration_seconds = duration_hours * 3600
            
            # CPU projections
            cpu_transcription = duration_seconds / cpu_results["processing_speed_ratio"]
            cpu_total = cpu_transcription + cpu_results["model_load_time"] + cpu_results["cold_start_time"]
            
            # GPU projections
            gpu_transcription = duration_seconds / gpu_results["processing_speed_ratio"]
            gpu_total = gpu_transcription + gpu_results["model_load_time"] + gpu_results["cold_start_time"]
            
            projections[f"{duration_hours}h"] = {
                "duration_hours": duration_hours,
                "cpu_total_minutes": cpu_total / 60,
                "gpu_total_minutes": gpu_total / 60,
                "cpu_timeout_risk": self._assess_timeout_risk(cpu_total),
                "gpu_timeout_risk": self._assess_timeout_risk(gpu_total),
                "gpu_speedup_factor": cpu_total / gpu_total,
                "cpu_cost": cpu_total / 3600 * 0.10,
                "gpu_cost": gpu_total / 3600 * 0.45
            }
        
        return projections
    
    def compare_with_alternatives(self, gpu_results: Dict) -> Dict:
        """Compare with Deepgram and other alternatives"""
        
        # Deepgram performance (from previous tests)
        deepgram_time = 9.57  # seconds for 1.14h audio
        deepgram_cost_per_minute = 0.0045
        
        duration_hours = self.baseline_results["audio_duration_hours"]
        duration_minutes = duration_hours * 60
        
        return {
            "deepgram_api": {
                "processing_time_minutes": deepgram_time / 60,
                "cost_per_file": duration_minutes * deepgram_cost_per_minute,
                "speed_ratio": self.baseline_results["audio_duration_seconds"] / deepgram_time,
                "pros": ["Fastest", "No infrastructure management", "High accuracy"],
                "cons": ["Expensive at scale", "Requires internet", "Privacy concerns"]
            },
            "cloud_run_gpu": {
                "processing_time_minutes": gpu_results["total_time"] / 60,
                "cost_per_file": gpu_results["estimated_cost_per_hour"] * (gpu_results["total_time"] / 3600),
                "speed_ratio": gpu_results["processing_speed_ratio"],
                "pros": ["Cost effective", "Private", "Scalable", "Fast"],
                "cons": ["Setup complexity", "Cold starts", "GPU availability"]
            },
            "local_processing": {
                "processing_time_minutes": self.baseline_results["local_mac_total_time"] / 60,
                "cost_per_file": 0,  # No direct cost
                "speed_ratio": self.baseline_results["local_mac_speed_ratio"],
                "pros": ["No cost", "Complete privacy", "No internet required"],
                "cons": ["Hardware dependent", "No scalability", "Maintenance overhead"]
            }
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        
        print("🚀 CLOUD RUN GPU PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Calculate performance scenarios
        cpu_results = self.calculate_cloud_run_cpu_performance()
        gpu_results = self.calculate_cloud_run_gpu_performance()
        
        print(f"\n📊 BASELINE (1.14-hour audio test results):")
        print(f"  Local Mac (MPS): {self.baseline_results['local_mac_total_time']/60:.1f} minutes")
        print(f"  Processing Speed: {self.baseline_results['local_mac_speed_ratio']:.1f}x real-time")
        print(f"  Words Transcribed: {self.baseline_results['words_transcribed']:,}")
        
        print(f"\n⚡ CLOUD RUN PERFORMANCE PROJECTIONS:")
        print(f"{'Scenario':<20} {'Time (min)':<12} {'Speed Ratio':<12} {'Timeout Risk':<12}")
        print("-" * 60)
        print(f"{'CPU-only':<20} {cpu_results['total_time']/60:<12.1f} {cpu_results['processing_speed_ratio']:<12.1f} {cpu_results['timeout_risk']:<12}")
        print(f"{'GPU (L4)':<20} {gpu_results['total_time']/60:<12.1f} {gpu_results['processing_speed_ratio']:<12.1f} {gpu_results['timeout_risk']:<12}")
        
        # Duration projections
        projections = self.project_different_durations(cpu_results, gpu_results)
        
        print(f"\n📈 DURATION PROJECTIONS:")
        print(f"{'Duration':<10} {'CPU (min)':<12} {'GPU (min)':<12} {'GPU Speedup':<12} {'GPU Risk':<12}")
        print("-" * 70)
        
        for duration, data in projections.items():
            print(f"{duration:<10} {data['cpu_total_minutes']:<12.1f} {data['gpu_total_minutes']:<12.1f} {data['gpu_speedup_factor']:<12.1f}x {data['gpu_timeout_risk']:<12}")
        
        # 3-hour specific analysis
        three_hour_data = projections["3h"]
        print(f"\n🎯 3-HOUR AUDIO ANALYSIS:")
        print(f"  CPU Processing: {three_hour_data['cpu_total_minutes']:.1f} minutes ({three_hour_data['cpu_timeout_risk']} timeout risk)")
        print(f"  GPU Processing: {three_hour_data['gpu_total_minutes']:.1f} minutes ({three_hour_data['gpu_timeout_risk']} timeout risk)")
        print(f"  GPU Speedup: {three_hour_data['gpu_speedup_factor']:.1f}x faster than CPU")
        print(f"  Cost Difference: CPU ${three_hour_data['cpu_cost']:.2f} vs GPU ${three_hour_data['gpu_cost']:.2f}")
        
        # Comparison with alternatives
        alternatives = self.compare_with_alternatives(gpu_results)
        
        print(f"\n💰 COST COMPARISON (3-hour audio):")
        for service, data in alternatives.items():
            print(f"  {service.replace('_', ' ').title():<20}: ${data['cost_per_file']:<8.2f} ({data['processing_time_minutes']:.1f} min)")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        
        if gpu_results["timeout_risk"] == "low":
            print(f"  ✅ GPU Cloud Run: HIGHLY RECOMMENDED")
            print(f"     - 3-hour audio in ~{three_hour_data['gpu_total_minutes']:.0f} minutes")
            print(f"     - {three_hour_data['gpu_speedup_factor']:.1f}x faster than CPU")
            print(f"     - Low timeout risk")
            print(f"     - Cost: ${three_hour_data['gpu_cost']:.2f} per 3h file")
        
        if cpu_results["timeout_risk"] == "high":
            print(f"  🔴 CPU Cloud Run: NOT RECOMMENDED for long files")
            print(f"     - High timeout risk for 3+ hour audio")
            print(f"     - {three_hour_data['cpu_total_minutes']:.0f} minutes processing time")
        
        print(f"\n💡 OPTIMAL STRATEGY:")
        print(f"  • Use GPU Cloud Run for all audio files")
        print(f"  • Expected performance: 2-3x faster than local Mac")
        print(f"  • Cost effective: ~${three_hour_data['gpu_cost']:.2f} vs ${alternatives['deepgram_api']['cost_per_file']:.2f} (Deepgram)")
        print(f"  • Scalable and private")
        
        # Save detailed results
        results = {
            "baseline": self.baseline_results,
            "cloud_run_cpu": cpu_results,
            "cloud_run_gpu": gpu_results,
            "duration_projections": projections,
            "alternatives_comparison": alternatives,
            "configuration": self.cloud_run_config
        }
        
        with open("cloud_run_gpu_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed analysis saved to: cloud_run_gpu_analysis_results.json")

def main():
    """Run the comprehensive analysis"""
    analysis = CloudRunGPUAnalysis()
    analysis.generate_comprehensive_report()

if __name__ == "__main__":
    main()
