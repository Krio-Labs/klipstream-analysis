#!/usr/bin/env python3
"""
Transcriber Comparison Test Script

This script compares the performance and output quality between:
1. Deepgram transcriber (current implementation)
2. Parakeet transcriber (new local implementation)

Usage:
    python test_transcriber_comparison.py <video_id> [--audio-file <path>]
"""

import asyncio
import argparse
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from raw_pipeline.transcriber import TranscriptionHandler as DeepgramTranscriber
from raw_pipeline.transcriber_parakeet import ParakeetTranscriptionHandler
from utils.logging_setup import setup_logger
from utils.config import RAW_TRANSCRIPTS_DIR

# Set up logger
logger = setup_logger("transcriber_comparison", "transcriber_comparison.log")

class TranscriberComparison:
    """Class for comparing transcriber performance and quality"""
    
    def __init__(self, video_id: str, audio_file_path: Optional[str] = None):
        self.video_id = video_id
        self.audio_file_path = audio_file_path
        self.results = {}
        
    async def run_deepgram_transcription(self) -> Dict:
        """Run Deepgram transcription and measure performance"""
        logger.info("🔵 Starting Deepgram transcription...")
        
        start_time = time.time()
        try:
            transcriber = DeepgramTranscriber()
            result = await transcriber.process_audio_files(
                self.video_id,
                self.audio_file_path,
                RAW_TRANSCRIPTS_DIR / "deepgram_test"
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            logger.info(f"✅ Deepgram transcription completed in {processing_time:.2f} seconds")
            
            return {
                "status": "success",
                "processing_time": processing_time,
                "files": result,
                "error": None
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.error(f"❌ Deepgram transcription failed: {str(e)}")
            
            return {
                "status": "failed",
                "processing_time": processing_time,
                "files": None,
                "error": str(e)
            }
    
    async def run_parakeet_transcription(self) -> Dict:
        """Run Parakeet transcription and measure performance"""
        logger.info("🟢 Starting Parakeet transcription...")
        
        start_time = time.time()
        try:
            transcriber = ParakeetTranscriptionHandler()
            result = await transcriber.process_audio_files(
                self.video_id,
                self.audio_file_path,
                RAW_TRANSCRIPTS_DIR / "parakeet_test"
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            logger.info(f"✅ Parakeet transcription completed in {processing_time:.2f} seconds")
            
            return {
                "status": "success",
                "processing_time": processing_time,
                "files": result,
                "error": None
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.error(f"❌ Parakeet transcription failed: {str(e)}")
            
            return {
                "status": "failed",
                "processing_time": processing_time,
                "files": None,
                "error": str(e)
            }
    
    def analyze_transcription_files(self, deepgram_files: Dict, parakeet_files: Dict) -> Dict:
        """Analyze and compare the transcription output files"""
        analysis = {
            "deepgram": {},
            "parakeet": {},
            "comparison": {}
        }
        
        try:
            # Analyze Deepgram files
            if deepgram_files and deepgram_files.get("words_file"):
                analysis["deepgram"] = self._analyze_transcription_output(deepgram_files, "Deepgram")
            
            # Analyze Parakeet files
            if parakeet_files and parakeet_files.get("words_file"):
                analysis["parakeet"] = self._analyze_transcription_output(parakeet_files, "Parakeet")
            
            # Compare outputs
            if analysis["deepgram"] and analysis["parakeet"]:
                analysis["comparison"] = self._compare_transcriptions(
                    analysis["deepgram"], 
                    analysis["parakeet"]
                )
            
        except Exception as e:
            logger.error(f"Error analyzing transcription files: {str(e)}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_transcription_output(self, files: Dict, transcriber_name: str) -> Dict:
        """Analyze transcription output files for metrics"""
        analysis = {"transcriber": transcriber_name}
        
        try:
            # Analyze words file
            words_file = files.get("words_file")
            if words_file and Path(words_file).exists():
                words_df = pd.read_csv(words_file)
                analysis["word_count"] = len(words_df)
                analysis["total_duration"] = words_df["end_time"].max() - words_df["start_time"].min()
                analysis["average_word_duration"] = words_df["end_time"].subtract(words_df["start_time"]).mean()
                
                # Get full transcript text
                analysis["full_transcript"] = " ".join(words_df["word"].tolist())
                analysis["character_count"] = len(analysis["full_transcript"])
            
            # Analyze paragraphs file
            paragraphs_file = files.get("paragraphs_file")
            if paragraphs_file and Path(paragraphs_file).exists():
                paragraphs_df = pd.read_csv(paragraphs_file)
                analysis["paragraph_count"] = len(paragraphs_df)
                analysis["average_paragraph_duration"] = paragraphs_df["end_time"].subtract(paragraphs_df["start_time"]).mean()
                analysis["average_paragraph_length"] = paragraphs_df["text"].str.len().mean()
            
            # File sizes
            for file_type, file_path in files.items():
                if file_path and Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    analysis[f"{file_type}_size_bytes"] = file_size
                    analysis[f"{file_type}_size_mb"] = file_size / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Error analyzing {transcriber_name} output: {str(e)}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _compare_transcriptions(self, deepgram_analysis: Dict, parakeet_analysis: Dict) -> Dict:
        """Compare the two transcription outputs"""
        comparison = {}
        
        try:
            # Word count comparison
            dg_words = deepgram_analysis.get("word_count", 0)
            pk_words = parakeet_analysis.get("word_count", 0)
            comparison["word_count_difference"] = pk_words - dg_words
            comparison["word_count_ratio"] = pk_words / dg_words if dg_words > 0 else 0
            
            # Duration comparison
            dg_duration = deepgram_analysis.get("total_duration", 0)
            pk_duration = parakeet_analysis.get("total_duration", 0)
            comparison["duration_difference"] = pk_duration - dg_duration
            
            # Character count comparison
            dg_chars = deepgram_analysis.get("character_count", 0)
            pk_chars = parakeet_analysis.get("character_count", 0)
            comparison["character_count_difference"] = pk_chars - dg_chars
            comparison["character_count_ratio"] = pk_chars / dg_chars if dg_chars > 0 else 0
            
            # Paragraph count comparison
            dg_paras = deepgram_analysis.get("paragraph_count", 0)
            pk_paras = parakeet_analysis.get("paragraph_count", 0)
            comparison["paragraph_count_difference"] = pk_paras - dg_paras
            
            # Simple text similarity (character-level)
            dg_text = deepgram_analysis.get("full_transcript", "").lower()
            pk_text = parakeet_analysis.get("full_transcript", "").lower()
            
            if dg_text and pk_text:
                # Calculate simple character overlap
                dg_chars_set = set(dg_text.replace(" ", ""))
                pk_chars_set = set(pk_text.replace(" ", ""))
                
                overlap = len(dg_chars_set.intersection(pk_chars_set))
                union = len(dg_chars_set.union(pk_chars_set))
                comparison["character_similarity"] = overlap / union if union > 0 else 0
                
                # Calculate word overlap
                dg_words_set = set(dg_text.split())
                pk_words_set = set(pk_text.split())
                
                word_overlap = len(dg_words_set.intersection(pk_words_set))
                word_union = len(dg_words_set.union(pk_words_set))
                comparison["word_similarity"] = word_overlap / word_union if word_union > 0 else 0
            
        except Exception as e:
            logger.error(f"Error comparing transcriptions: {str(e)}")
            comparison["error"] = str(e)
        
        return comparison
    
    async def run_comparison(self) -> Dict:
        """Run the full comparison between both transcribers"""
        logger.info(f"🚀 Starting transcriber comparison for video ID: {self.video_id}")
        
        # Create output directories
        (RAW_TRANSCRIPTS_DIR / "deepgram_test").mkdir(exist_ok=True, parents=True)
        (RAW_TRANSCRIPTS_DIR / "parakeet_test").mkdir(exist_ok=True, parents=True)
        
        # Run both transcriptions
        deepgram_result = await self.run_deepgram_transcription()
        parakeet_result = await self.run_parakeet_transcription()
        
        # Analyze results
        analysis = {}
        if (deepgram_result["status"] == "success" and 
            parakeet_result["status"] == "success"):
            analysis = self.analyze_transcription_files(
                deepgram_result["files"],
                parakeet_result["files"]
            )
        
        # Compile final results
        results = {
            "video_id": self.video_id,
            "audio_file_path": self.audio_file_path,
            "deepgram_result": deepgram_result,
            "parakeet_result": parakeet_result,
            "analysis": analysis,
            "summary": {
                "deepgram_success": deepgram_result["status"] == "success",
                "parakeet_success": parakeet_result["status"] == "success",
                "deepgram_time": deepgram_result["processing_time"],
                "parakeet_time": parakeet_result["processing_time"],
                "time_difference": parakeet_result["processing_time"] - deepgram_result["processing_time"],
                "parakeet_faster": parakeet_result["processing_time"] < deepgram_result["processing_time"]
            }
        }
        
        # Save results to file
        results_file = RAW_TRANSCRIPTS_DIR / f"comparison_results_{self.video_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Comparison results saved to: {results_file}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print a summary of the comparison results"""
        print("\n" + "="*60)
        print("TRANSCRIBER COMPARISON SUMMARY")
        print("="*60)
        
        summary = results["summary"]
        
        print(f"Video ID: {results['video_id']}")
        print(f"Audio File: {results['audio_file_path'] or 'Auto-detected'}")
        print()
        
        print("PROCESSING RESULTS:")
        print(f"  Deepgram: {'✅ Success' if summary['deepgram_success'] else '❌ Failed'}")
        print(f"  Parakeet: {'✅ Success' if summary['parakeet_success'] else '❌ Failed'}")
        print()
        
        if summary["deepgram_success"] and summary["parakeet_success"]:
            print("PROCESSING TIME:")
            print(f"  Deepgram: {summary['deepgram_time']:.2f} seconds")
            print(f"  Parakeet: {summary['parakeet_time']:.2f} seconds")
            print(f"  Difference: {summary['time_difference']:.2f} seconds")
            print(f"  Faster: {'Parakeet' if summary['parakeet_faster'] else 'Deepgram'}")
            print()
            
            analysis = results.get("analysis", {})
            if analysis.get("comparison"):
                comp = analysis["comparison"]
                print("OUTPUT COMPARISON:")
                print(f"  Word count ratio (Parakeet/Deepgram): {comp.get('word_count_ratio', 0):.2f}")
                print(f"  Character similarity: {comp.get('character_similarity', 0):.2f}")
                print(f"  Word similarity: {comp.get('word_similarity', 0):.2f}")
        
        print("="*60)

async def main():
    """Main function to run the transcriber comparison"""
    parser = argparse.ArgumentParser(description="Compare Deepgram and Parakeet transcribers")
    parser.add_argument("video_id", help="Video ID to process")
    parser.add_argument("--audio-file", help="Path to audio file (optional)")
    
    args = parser.parse_args()
    
    comparison = TranscriberComparison(args.video_id, args.audio_file)
    await comparison.run_comparison()

if __name__ == "__main__":
    asyncio.run(main())
