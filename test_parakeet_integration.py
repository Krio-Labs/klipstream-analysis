#!/usr/bin/env python3
"""
Parakeet Integration Test Script

This script tests the Parakeet transcriber as a drop-in replacement for the Deepgram transcriber
in the existing pipeline architecture.

Usage:
    python test_parakeet_integration.py <video_id> [--audio-file <path>]
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from raw_pipeline.transcriber_parakeet import ParakeetTranscriptionHandler
from utils.logging_setup import setup_logger
from utils.config import RAW_TRANSCRIPTS_DIR, RAW_AUDIO_DIR

# Set up logger
logger = setup_logger("parakeet_integration_test", "parakeet_integration_test.log")

async def test_parakeet_integration(video_id: str, audio_file_path: str = None):
    """Test the Parakeet transcriber integration"""
    
    logger.info(f"🧪 Testing Parakeet transcriber integration for video ID: {video_id}")
    
    try:
        # Initialize the Parakeet transcriber
        logger.info("Initializing Parakeet transcriber...")
        transcriber = ParakeetTranscriptionHandler()
        logger.info("✅ Parakeet transcriber initialized successfully")
        
        # Test the process_audio_files method (same interface as Deepgram)
        logger.info("Testing process_audio_files method...")
        result = await transcriber.process_audio_files(
            video_id=video_id,
            audio_file_path=audio_file_path,
            output_dir=RAW_TRANSCRIPTS_DIR / "parakeet_integration_test"
        )
        
        if result is None:
            logger.error("❌ Transcription failed - no result returned")
            return False
        
        logger.info("✅ process_audio_files completed successfully")
        
        # Verify the output structure matches expected format
        expected_keys = ["paragraphs_file", "words_file", "transcript_json_file"]
        missing_keys = [key for key in expected_keys if key not in result]
        
        if missing_keys:
            logger.error(f"❌ Missing expected keys in result: {missing_keys}")
            return False
        
        logger.info("✅ Result structure matches expected format")
        
        # Verify all output files exist
        for file_type, file_path in result.items():
            if not Path(file_path).exists():
                logger.error(f"❌ Output file does not exist: {file_path}")
                return False
            else:
                file_size = Path(file_path).stat().st_size
                logger.info(f"✅ {file_type}: {file_path} ({file_size} bytes)")
        
        # Verify CSV file formats
        logger.info("Verifying CSV file formats...")
        
        # Check words file format
        words_file = result["words_file"]
        try:
            import pandas as pd
            words_df = pd.read_csv(words_file)
            expected_columns = ["start_time", "end_time", "word"]
            
            if list(words_df.columns) != expected_columns:
                logger.error(f"❌ Words file has incorrect columns: {list(words_df.columns)}")
                logger.error(f"   Expected: {expected_columns}")
                return False
            
            logger.info(f"✅ Words file format correct ({len(words_df)} words)")
            
        except Exception as e:
            logger.error(f"❌ Error reading words file: {str(e)}")
            return False
        
        # Check paragraphs file format
        paragraphs_file = result["paragraphs_file"]
        try:
            paragraphs_df = pd.read_csv(paragraphs_file)
            expected_columns = ["start_time", "end_time", "text"]
            
            if list(paragraphs_df.columns) != expected_columns:
                logger.error(f"❌ Paragraphs file has incorrect columns: {list(paragraphs_df.columns)}")
                logger.error(f"   Expected: {expected_columns}")
                return False
            
            logger.info(f"✅ Paragraphs file format correct ({len(paragraphs_df)} paragraphs)")
            
        except Exception as e:
            logger.error(f"❌ Error reading paragraphs file: {str(e)}")
            return False
        
        # Verify JSON file format
        transcript_json_file = result["transcript_json_file"]
        try:
            import json
            with open(transcript_json_file, 'r') as f:
                transcript_data = json.load(f)
            
            # Check for expected structure (mimicking Deepgram format)
            if "results" not in transcript_data:
                logger.error("❌ JSON file missing 'results' key")
                return False
            
            if "channels" not in transcript_data["results"]:
                logger.error("❌ JSON file missing 'results.channels' key")
                return False
            
            logger.info("✅ JSON file format correct")
            
        except Exception as e:
            logger.error(f"❌ Error reading JSON file: {str(e)}")
            return False
        
        # Test data consistency
        logger.info("Testing data consistency...")
        
        # Check that word count matches between files
        json_word_count = len(transcript_data["results"]["channels"][0]["alternatives"][0]["words"])
        csv_word_count = len(words_df)
        
        if json_word_count != csv_word_count:
            logger.warning(f"⚠️  Word count mismatch: JSON={json_word_count}, CSV={csv_word_count}")
        else:
            logger.info(f"✅ Word count consistent: {json_word_count} words")
        
        # Check that timestamps are reasonable
        if len(words_df) > 0:
            min_time = words_df["start_time"].min()
            max_time = words_df["end_time"].max()
            duration = max_time - min_time
            
            if min_time < 0:
                logger.error(f"❌ Negative start time found: {min_time}")
                return False
            
            if duration <= 0:
                logger.error(f"❌ Invalid duration: {duration}")
                return False
            
            logger.info(f"✅ Timestamps look reasonable: {duration:.2f} seconds total")
        
        logger.info("🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def find_audio_file(video_id: str) -> str:
    """Find an audio file for the given video ID"""
    possible_paths = [
        RAW_AUDIO_DIR / f"audio_{video_id}.mp3",
        RAW_AUDIO_DIR / f"audio_{video_id}.wav",
        Path(f"output/raw/audio_{video_id}.mp3"),
        Path(f"output/raw/audio_{video_id}.wav"),
        Path(f"outputs/audio_{video_id}.mp3"),
        Path(f"outputs/audio_{video_id}.wav"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

async def main():
    """Main function to run the integration test"""
    parser = argparse.ArgumentParser(description="Test Parakeet transcriber integration")
    parser.add_argument("video_id", help="Video ID to process")
    parser.add_argument("--audio-file", help="Path to audio file (optional)")
    
    args = parser.parse_args()
    
    # If no audio file specified, try to find one
    audio_file = args.audio_file
    if not audio_file:
        audio_file = find_audio_file(args.video_id)
        if not audio_file:
            logger.error(f"❌ No audio file found for video ID: {args.video_id}")
            logger.error("Please specify an audio file with --audio-file or ensure audio file exists in standard locations")
            sys.exit(1)
        else:
            logger.info(f"Found audio file: {audio_file}")
    
    # Run the integration test
    success = await test_parakeet_integration(args.video_id, audio_file)
    
    if success:
        print("\n🎉 Integration test PASSED! Parakeet transcriber is ready for use.")
        sys.exit(0)
    else:
        print("\n❌ Integration test FAILED! Please check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
