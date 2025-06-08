#!/usr/bin/env python3
"""
Audio Format Conversion Test Script

This script tests the audio format conversion functionality of the Parakeet transcriber
to ensure MP3 files are properly converted to WAV/FLAC for model compatibility.

Usage:
    python test_audio_conversion.py <audio_file_path>
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from raw_pipeline.transcriber_parakeet import ParakeetTranscriptionHandler
from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("audio_conversion_test", "audio_conversion_test.log")

def analyze_audio_file(file_path: str) -> dict:
    """Analyze an audio file and return its properties"""
    try:
        path = Path(file_path)
        
        # Basic file info
        info = {
            "file_path": str(path),
            "file_size_mb": path.stat().st_size / (1024 * 1024),
            "file_extension": path.suffix.lower(),
            "exists": path.exists()
        }
        
        if not path.exists():
            return info
        
        # Try to load with pydub first (handles more formats)
        try:
            audio_segment = AudioSegment.from_file(file_path)
            info.update({
                "pydub_duration_s": len(audio_segment) / 1000.0,
                "pydub_channels": audio_segment.channels,
                "pydub_sample_rate": audio_segment.frame_rate,
                "pydub_sample_width": audio_segment.sample_width
            })
        except Exception as e:
            info["pydub_error"] = str(e)
        
        # Try to load with librosa
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[0]
            
            info.update({
                "librosa_duration_s": len(audio_data) / sample_rate if audio_data.ndim == 1 else len(audio_data[0]) / sample_rate,
                "librosa_channels": channels,
                "librosa_sample_rate": sample_rate,
                "librosa_dtype": str(audio_data.dtype)
            })
        except Exception as e:
            info["librosa_error"] = str(e)
        
        return info
        
    except Exception as e:
        return {"error": str(e), "file_path": file_path}

def test_audio_conversion(input_file: str):
    """Test the audio conversion functionality"""
    
    logger.info(f"🧪 Testing audio conversion for: {input_file}")
    
    # Analyze original file
    logger.info("📊 Analyzing original audio file...")
    original_info = analyze_audio_file(input_file)
    
    print("\n" + "="*60)
    print("ORIGINAL AUDIO FILE ANALYSIS")
    print("="*60)
    for key, value in original_info.items():
        print(f"{key}: {value}")
    
    if not original_info.get("exists", False):
        logger.error(f"❌ Input file does not exist: {input_file}")
        return False
    
    # Test the Parakeet transcriber's conversion functionality
    try:
        logger.info("🔄 Testing Parakeet transcriber audio conversion...")
        transcriber = ParakeetTranscriptionHandler()
        
        # Test the conversion method directly
        input_path = Path(input_file)
        temp_output_path = input_path.parent / f"{input_path.stem}_test_converted.wav"
        
        # Only test conversion if the file is not already WAV/FLAC
        if input_path.suffix.lower() not in [".wav", ".flac"]:
            logger.info(f"Converting {input_path.suffix} to WAV...")
            converted_path = transcriber._convert_audio_format(
                str(input_path),
                str(temp_output_path),
                "wav"
            )
            
            # Analyze converted file
            logger.info("📊 Analyzing converted audio file...")
            converted_info = analyze_audio_file(converted_path)
            
            print("\n" + "="*60)
            print("CONVERTED AUDIO FILE ANALYSIS")
            print("="*60)
            for key, value in converted_info.items():
                print(f"{key}: {value}")
            
            # Test loading the converted file with the transcriber's method
            logger.info("🔍 Testing audio loading with transcriber...")
            audio_data, sample_rate = transcriber._load_and_preprocess_audio(input_file)
            
            print("\n" + "="*60)
            print("TRANSCRIBER AUDIO LOADING RESULTS")
            print("="*60)
            print(f"Audio shape: {audio_data.shape}")
            print(f"Sample rate: {sample_rate}")
            print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
            print(f"Data type: {audio_data.dtype}")
            print(f"Value range: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
            
            # Test audio validation
            logger.info("✅ Testing audio validation...")
            is_valid = transcriber._validate_audio_format(audio_data, sample_rate)
            print(f"Audio validation result: {'✅ PASSED' if is_valid else '❌ FAILED'}")
            
            # Clean up temporary file
            if Path(converted_path).exists():
                Path(converted_path).unlink()
                logger.info("🧹 Cleaned up temporary converted file")
            
        else:
            logger.info(f"File is already in compatible format ({input_path.suffix})")
            
            # Test loading directly
            logger.info("🔍 Testing direct audio loading...")
            audio_data, sample_rate = transcriber._load_and_preprocess_audio(input_file)
            
            print("\n" + "="*60)
            print("DIRECT AUDIO LOADING RESULTS")
            print("="*60)
            print(f"Audio shape: {audio_data.shape}")
            print(f"Sample rate: {sample_rate}")
            print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
            print(f"Data type: {audio_data.dtype}")
            print(f"Value range: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
            
            # Test audio validation
            is_valid = transcriber._validate_audio_format(audio_data, sample_rate)
            print(f"Audio validation result: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        logger.info("✅ Audio conversion test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Audio conversion test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the audio conversion test"""
    parser = argparse.ArgumentParser(description="Test audio format conversion for Parakeet transcriber")
    parser.add_argument("audio_file", help="Path to audio file to test")
    
    args = parser.parse_args()
    
    if not Path(args.audio_file).exists():
        logger.error(f"❌ Audio file does not exist: {args.audio_file}")
        sys.exit(1)
    
    success = test_audio_conversion(args.audio_file)
    
    if success:
        print("\n🎉 Audio conversion test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Audio conversion test FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
