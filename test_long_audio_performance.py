#!/usr/bin/env python3
"""
Long Audio Performance Test Script

This script tests the Parakeet transcriber performance with long audio files (1+ hours)
and provides detailed performance metrics including memory usage, processing speed,
and chunking efficiency.

Usage:
    python test_long_audio_performance.py <audio_file_path> [--deepgram-compare]
"""

import asyncio
import argparse
import time
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import json

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from raw_pipeline.transcriber_parakeet import ParakeetTranscriptionHandler
from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("long_audio_test", "long_audio_test.log")

class LongAudioPerformanceTest:
    """Class for testing long audio file performance"""
    
    def __init__(self, audio_file_path: str):
        self.audio_file_path = audio_file_path
        self.audio_file_size = 0
        self.audio_duration = 0
        self.performance_metrics = {}
        
    def get_audio_info(self):
        """Get basic audio file information"""
        try:
            from pydub import AudioSegment
            
            # File size
            self.audio_file_size = os.path.getsize(self.audio_file_path) / (1024 * 1024)  # MB
            
            # Audio duration
            audio = AudioSegment.from_file(self.audio_file_path)
            self.audio_duration = len(audio) / 1000.0  # seconds
            
            logger.info(f"Audio file: {self.audio_file_path}")
            logger.info(f"File size: {self.audio_file_size:.2f} MB")
            logger.info(f"Duration: {self.audio_duration:.2f} seconds ({self.audio_duration/3600:.2f} hours)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return False
    
    def monitor_system_resources(self):
        """Get current system resource usage"""
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "memory_percent": process.memory_percent(),
            "available_memory_gb": psutil.virtual_memory().available / (1024 * 1024 * 1024)
        }
    
    async def test_parakeet_performance(self, output_dir: str = "output/long_audio_test") -> Dict:
        """Test Parakeet transcriber with long audio file"""
        
        print(f"\n🚀 Starting Parakeet performance test...")
        print(f"📁 Audio file: {self.audio_file_path}")
        print(f"📊 File size: {self.audio_file_size:.2f} MB")
        print(f"⏱️  Duration: {self.audio_duration/3600:.2f} hours")
        
        # Initial system state
        initial_resources = self.monitor_system_resources()
        print(f"💾 Initial memory usage: {initial_resources['memory_mb']:.1f} MB")
        print(f"🖥️  Available memory: {initial_resources['available_memory_gb']:.1f} GB")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Initialize transcriber
            print("\n🔧 Initializing Parakeet transcriber...")
            init_start = time.time()
            transcriber = ParakeetTranscriptionHandler()
            init_time = time.time() - init_start
            
            post_init_resources = self.monitor_system_resources()
            print(f"✅ Model loaded in {init_time:.2f} seconds")
            print(f"💾 Memory after model load: {post_init_resources['memory_mb']:.1f} MB")
            
            # Run transcription
            print("\n🎵 Starting transcription...")
            transcription_start = time.time()
            
            result = await transcriber.process_audio_files(
                video_id="long_audio_test",
                audio_file_path=self.audio_file_path,
                output_dir=output_dir
            )
            
            transcription_time = time.time() - transcription_start
            total_time = time.time() - start_time
            
            # Final system state
            final_resources = self.monitor_system_resources()
            
            if result:
                print(f"\n✅ Transcription completed successfully!")
                
                # Analyze output files
                words_df = pd.read_csv(result['words_file'])
                paragraphs_df = pd.read_csv(result['paragraphs_file'])
                
                # Calculate performance metrics
                processing_speed_ratio = self.audio_duration / transcription_time
                words_per_second = len(words_df) / transcription_time
                
                performance_data = {
                    "status": "success",
                    "audio_duration_hours": self.audio_duration / 3600,
                    "audio_duration_seconds": self.audio_duration,
                    "file_size_mb": self.audio_file_size,
                    "model_load_time": init_time,
                    "transcription_time": transcription_time,
                    "total_time": total_time,
                    "processing_speed_ratio": processing_speed_ratio,
                    "words_transcribed": len(words_df),
                    "paragraphs_created": len(paragraphs_df),
                    "words_per_second": words_per_second,
                    "memory_usage": {
                        "initial_mb": initial_resources['memory_mb'],
                        "post_init_mb": post_init_resources['memory_mb'],
                        "final_mb": final_resources['memory_mb'],
                        "peak_increase_mb": final_resources['memory_mb'] - initial_resources['memory_mb']
                    },
                    "output_files": result
                }
                
                # Print detailed results
                print(f"\n📊 PERFORMANCE RESULTS:")
                print(f"⏱️  Model loading: {init_time:.2f} seconds")
                print(f"⏱️  Transcription: {transcription_time:.2f} seconds ({transcription_time/60:.1f} minutes)")
                print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
                print(f"🚀 Processing speed: {processing_speed_ratio:.2f}x real-time")
                print(f"📝 Words transcribed: {len(words_df):,}")
                print(f"📄 Paragraphs created: {len(paragraphs_df):,}")
                print(f"⚡ Words per second: {words_per_second:.1f}")
                print(f"💾 Memory increase: {final_resources['memory_mb'] - initial_resources['memory_mb']:.1f} MB")
                
                # Estimate for different durations
                print(f"\n📈 SCALING ESTIMATES:")
                print(f"1 hour audio: ~{transcription_time * (3600/self.audio_duration)/60:.1f} minutes")
                print(f"3 hour audio: ~{transcription_time * (10800/self.audio_duration)/60:.1f} minutes")
                print(f"10 hour audio: ~{transcription_time * (36000/self.audio_duration)/60:.1f} minutes")
                
                return performance_data
                
            else:
                print("❌ Transcription failed")
                return {
                    "status": "failed",
                    "error": "Transcription returned None",
                    "total_time": total_time
                }
                
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ Error during transcription: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_time": total_time
            }
    
    async def compare_with_deepgram(self, output_dir: str = "output/long_audio_test") -> Dict:
        """Compare Parakeet with Deepgram for long audio"""
        
        print(f"\n🔵 Testing Deepgram for comparison...")
        
        try:
            from raw_pipeline.transcriber import TranscriptionHandler as DeepgramTranscriber
            
            start_time = time.time()
            initial_resources = self.monitor_system_resources()
            
            transcriber = DeepgramTranscriber()
            result = await transcriber.process_audio_files(
                video_id="long_audio_test_deepgram",
                audio_file_path=self.audio_file_path,
                output_dir=f"{output_dir}/deepgram"
            )
            
            total_time = time.time() - start_time
            final_resources = self.monitor_system_resources()
            
            if result:
                words_df = pd.read_csv(result['words_file'])
                paragraphs_df = pd.read_csv(result['paragraphs_file'])
                
                return {
                    "status": "success",
                    "transcription_time": total_time,
                    "words_transcribed": len(words_df),
                    "paragraphs_created": len(paragraphs_df),
                    "processing_speed_ratio": self.audio_duration / total_time,
                    "memory_increase_mb": final_resources['memory_mb'] - initial_resources['memory_mb']
                }
            else:
                return {"status": "failed", "transcription_time": total_time}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def save_results(self, parakeet_results: Dict, deepgram_results: Dict = None):
        """Save test results to file"""
        
        results = {
            "test_info": {
                "audio_file": self.audio_file_path,
                "file_size_mb": self.audio_file_size,
                "duration_hours": self.audio_duration / 3600,
                "duration_seconds": self.audio_duration,
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "parakeet_results": parakeet_results
        }
        
        if deepgram_results:
            results["deepgram_results"] = deepgram_results
            
            # Add comparison
            if (parakeet_results.get("status") == "success" and 
                deepgram_results.get("status") == "success"):
                
                results["comparison"] = {
                    "speed_ratio": deepgram_results["transcription_time"] / parakeet_results["transcription_time"],
                    "parakeet_faster": parakeet_results["transcription_time"] < deepgram_results["transcription_time"],
                    "word_count_ratio": parakeet_results["words_transcribed"] / deepgram_results["words_transcribed"],
                    "memory_difference_mb": parakeet_results["memory_usage"]["peak_increase_mb"] - deepgram_results["memory_increase_mb"]
                }
        
        # Save to file
        results_file = f"long_audio_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        return results_file

async def main():
    """Main function to run the long audio performance test"""
    parser = argparse.ArgumentParser(description="Test Parakeet transcriber with long audio files")
    parser.add_argument("audio_file", help="Path to long audio file")
    parser.add_argument("--deepgram-compare", action="store_true", help="Also test with Deepgram for comparison")
    parser.add_argument("--output-dir", default="output/long_audio_test", help="Output directory for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize test
    test = LongAudioPerformanceTest(args.audio_file)
    
    # Get audio info
    if not test.get_audio_info():
        print("❌ Failed to analyze audio file")
        sys.exit(1)
    
    # Warn if file is very long
    if test.audio_duration > 7200:  # 2 hours
        print(f"\n⚠️  WARNING: This is a {test.audio_duration/3600:.1f} hour audio file.")
        print("This test may take a very long time and use significant memory.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Test cancelled.")
            sys.exit(0)
    
    # Run Parakeet test
    parakeet_results = await test.test_parakeet_performance(args.output_dir)
    
    # Run Deepgram comparison if requested
    deepgram_results = None
    if args.deepgram_compare:
        deepgram_results = await test.compare_with_deepgram(args.output_dir)
    
    # Save and display results
    results_file = test.save_results(parakeet_results, deepgram_results)
    
    print(f"\n🎉 Long audio performance test completed!")
    print(f"📄 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main())
