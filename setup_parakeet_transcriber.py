#!/usr/bin/env python3
"""
Parakeet Transcriber Setup Script

This script helps set up the Parakeet transcriber by:
1. Installing required dependencies
2. Downloading the model
3. Testing the installation
4. Providing usage examples

Usage:
    python setup_parakeet_transcriber.py [--test-audio <path>]
"""

import subprocess
import sys
import argparse
from pathlib import Path
import importlib.util

def check_package_installed(package_name):
    """Check if a Python package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_dependencies():
    """Install required dependencies for Parakeet transcriber"""
    print("🔧 Installing Parakeet transcriber dependencies...")
    
    required_packages = [
        "transformers>=4.36.0",
        "torch>=2.1.0", 
        "torchaudio>=2.1.0",
        "accelerate>=0.25.0"
    ]
    
    # Check which packages are missing
    missing_packages = []
    for package in required_packages:
        package_name = package.split(">=")[0].split("==")[0]
        if not check_package_installed(package_name):
            missing_packages.append(package)
    
    if not missing_packages:
        print("✅ All required packages are already installed!")
        return True
    
    print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
    
    try:
        # Install missing packages
        for package in missing_packages:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        
        print("✅ All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_model():
    """Download the Parakeet model"""
    print("📥 Downloading Parakeet TDT 0.6B v2 model...")
    
    try:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        model_name = "nvidia/parakeet-tdt-0.6b-v2"
        
        print(f"Downloading processor for {model_name}...")
        processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"Downloading model {model_name}...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        print("✅ Model downloaded successfully!")
        print(f"📊 Model size: ~2.4GB")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def test_installation(test_audio_path=None):
    """Test the Parakeet transcriber installation"""
    print("🧪 Testing Parakeet transcriber installation...")
    
    try:
        # Import the transcriber
        from raw_pipeline.transcriber_parakeet import ParakeetTranscriptionHandler
        
        print("✅ Successfully imported ParakeetTranscriptionHandler")
        
        # Initialize the transcriber
        print("Initializing transcriber...")
        transcriber = ParakeetTranscriptionHandler()
        
        print("✅ Transcriber initialized successfully!")
        print(f"Using device: {transcriber.device}")
        
        # If test audio provided, test transcription
        if test_audio_path and Path(test_audio_path).exists():
            print(f"🎵 Testing transcription with: {test_audio_path}")
            
            import asyncio
            
            async def test_transcription():
                result = await transcriber.process_audio_files(
                    video_id="test",
                    audio_file_path=test_audio_path,
                    output_dir=Path("output/test_parakeet")
                )
                return result
            
            result = asyncio.run(test_transcription())
            
            if result:
                print("✅ Test transcription completed successfully!")
                for file_type, file_path in result.items():
                    if Path(file_path).exists():
                        file_size = Path(file_path).stat().st_size
                        print(f"  {file_type}: {file_path} ({file_size} bytes)")
            else:
                print("❌ Test transcription failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def print_usage_examples():
    """Print usage examples for the Parakeet transcriber"""
    print("\n" + "="*60)
    print("PARAKEET TRANSCRIBER USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Basic Usage (Drop-in replacement):")
    print("""
from raw_pipeline.transcriber_parakeet import ParakeetTranscriptionHandler

# Initialize transcriber
transcriber = ParakeetTranscriptionHandler()

# Process audio file
result = await transcriber.process_audio_files(
    video_id="12345",
    audio_file_path="/path/to/audio.mp3"
)
""")
    
    print("\n2. Test Scripts:")
    print("""
# Compare Deepgram vs Parakeet
python test_transcriber_comparison.py 12345 --audio-file /path/to/audio.mp3

# Test integration
python test_parakeet_integration.py 12345 --audio-file /path/to/audio.mp3

# Test audio conversion
python test_audio_conversion.py /path/to/audio.mp3
""")
    
    print("\n3. Pipeline Integration:")
    print("""
# Option 1: Direct replacement in processor.py
from .transcriber_parakeet import ParakeetTranscriptionHandler as TranscriptionHandler

# Option 2: Environment variable control
USE_LOCAL_TRANSCRIPTION=true python main.py <video_url>
""")
    
    print("\n4. Performance Tips:")
    print("""
- Use GPU for faster processing (automatically detected)
- For large files, processing happens in 30-second chunks
- MP3 files are automatically converted to WAV
- Model is cached after first load (~2.4GB)
""")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Parakeet transcriber")
    parser.add_argument("--test-audio", help="Path to audio file for testing")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download")
    parser.add_argument("--skip-test", action="store_true", help="Skip installation test")
    
    args = parser.parse_args()
    
    print("🚀 Setting up Parakeet Transcriber...")
    print("="*50)
    
    success = True
    
    # Install dependencies
    if not args.skip_install:
        if not install_dependencies():
            success = False
    
    # Download model
    if success and not args.skip_download:
        if not download_model():
            success = False
    
    # Test installation
    if success and not args.skip_test:
        if not test_installation(args.test_audio):
            success = False
    
    if success:
        print("\n🎉 Parakeet transcriber setup completed successfully!")
        print_usage_examples()
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
