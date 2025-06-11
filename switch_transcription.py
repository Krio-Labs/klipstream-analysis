#!/usr/bin/env python3
"""
Quick Transcription Method Switcher

This script provides an easy way to switch between Deepgram and GPU transcription
for the klipstream-analysis pipeline.

Usage:
    python switch_transcription.py deepgram    # Switch to Deepgram
    python switch_transcription.py gpu         # Switch to GPU transcription
    python switch_transcription.py status      # Show current status
    python switch_transcription.py toggle      # Toggle between methods
"""

import sys
import os
from pathlib import Path

def main():
    """Main function for transcription switching"""
    
    # Add current directory to path to import transcription_config
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        from transcription_config import (
            enable_deepgram,
            enable_gpu_transcription,
            toggle_transcription_method,
            print_transcription_status,
            get_transcription_config
        )
    except ImportError as e:
        print(f"❌ Error importing transcription_config: {e}")
        print("Make sure transcription_config.py is in the same directory")
        return 1
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("🎤 TRANSCRIPTION METHOD SWITCHER")
        print("=" * 50)
        print("Usage: python switch_transcription.py [command]")
        print("")
        print("Commands:")
        print("  deepgram  - Switch to Deepgram transcription (temporary)")
        print("  gpu       - Switch to GPU transcription (original)")
        print("  status    - Show current transcription configuration")
        print("  toggle    - Toggle between Deepgram and GPU")
        print("")
        print("Current status:")
        print_transcription_status()
        return 0
    
    command = sys.argv[1].lower()
    
    if command in ["deepgram", "dg", "cloud"]:
        print("🔄 Switching to Deepgram transcription...")
        enable_deepgram()
        print("")
        print_transcription_status()
        print("")
        print("✅ Deepgram transcription enabled!")
        print("💡 This is temporary - GPU transcription will be re-enabled soon")
        
    elif command in ["gpu", "parakeet", "local"]:
        print("🔄 Switching to GPU transcription...")
        enable_gpu_transcription()
        print("")
        print_transcription_status()
        print("")
        print("✅ GPU transcription enabled!")
        print("💡 Make sure your system has compatible GPU hardware")
        
    elif command in ["status", "show", "current", "info"]:
        print_transcription_status()
        
    elif command in ["toggle", "switch", "flip"]:
        config = get_transcription_config()
        current_method = "Deepgram" if config['using_deepgram'] else "GPU"
        new_method = "GPU" if config['using_deepgram'] else "Deepgram"
        
        print(f"🔄 Toggling from {current_method} to {new_method} transcription...")
        toggle_transcription_method()
        print("")
        print_transcription_status()
        
    elif command in ["help", "-h", "--help"]:
        print("🎤 TRANSCRIPTION METHOD SWITCHER - HELP")
        print("=" * 50)
        print("")
        print("This tool helps you switch between transcription methods:")
        print("")
        print("🌐 DEEPGRAM TRANSCRIPTION:")
        print("   - Cloud-based transcription service")
        print("   - Reliable and fast")
        print("   - Currently used as temporary solution")
        print("   - Requires API key (DEEPGRAM_API_KEY)")
        print("")
        print("🖥️  GPU TRANSCRIPTION:")
        print("   - Local GPU-based transcription")
        print("   - Uses NVIDIA Parakeet models")
        print("   - Cost-effective for high volume")
        print("   - Requires compatible GPU hardware")
        print("")
        print("COMMANDS:")
        print("   deepgram  - Switch to Deepgram (temporary)")
        print("   gpu       - Switch to GPU transcription")
        print("   status    - Show current configuration")
        print("   toggle    - Toggle between methods")
        print("   help      - Show this help message")
        print("")
        print("ENVIRONMENT VARIABLES:")
        print("   FORCE_DEEPGRAM_TRANSCRIPTION=true/false")
        print("   DEEPGRAM_API_KEY=your_api_key")
        print("")
        print("EXAMPLES:")
        print("   python switch_transcription.py deepgram")
        print("   python switch_transcription.py gpu")
        print("   python switch_transcription.py status")
        print("   FORCE_DEEPGRAM_TRANSCRIPTION=false python main.py")
        
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python switch_transcription.py help' for usage information")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
