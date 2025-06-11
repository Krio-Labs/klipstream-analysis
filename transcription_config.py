#!/usr/bin/env python3
"""
Transcription Configuration Management

This module provides easy configuration switching between Deepgram and GPU transcription.
Designed for temporary use while GPU transcription is being optimized.
"""

import os
from typing import Dict, Any

# =============================================================================
# TRANSCRIPTION CONFIGURATION
# =============================================================================

# TEMPORARY CONFIGURATION: Set to True to use Deepgram, False to use GPU transcription
USE_DEEPGRAM_TEMPORARILY = True

# Alternative ways to control this:
# 1. Environment variable: FORCE_DEEPGRAM_TRANSCRIPTION=true/false
# 2. Change the constant above
# 3. Call set_transcription_method() programmatically

def get_transcription_method() -> str:
    """
    Get the current transcription method based on configuration
    
    Returns:
        str: 'deepgram' or 'auto' (for GPU transcription)
    """
    # Check environment variable first (highest priority)
    env_force_deepgram = os.environ.get("FORCE_DEEPGRAM_TRANSCRIPTION", "").lower()
    if env_force_deepgram in ["true", "1", "yes"]:
        return "deepgram"
    elif env_force_deepgram in ["false", "0", "no"]:
        return "auto"  # Will use GPU transcription if available
    
    # Fall back to module constant
    if USE_DEEPGRAM_TEMPORARILY:
        return "deepgram"
    else:
        return "auto"

def set_transcription_method(use_deepgram: bool = True):
    """
    Programmatically set the transcription method
    
    Args:
        use_deepgram (bool): True to use Deepgram, False to use GPU transcription
    """
    global USE_DEEPGRAM_TEMPORARILY
    USE_DEEPGRAM_TEMPORARILY = use_deepgram
    
    # Also set environment variable for consistency
    os.environ["FORCE_DEEPGRAM_TRANSCRIPTION"] = "true" if use_deepgram else "false"

def is_using_deepgram() -> bool:
    """
    Check if currently configured to use Deepgram
    
    Returns:
        bool: True if using Deepgram, False if using GPU transcription
    """
    return get_transcription_method() == "deepgram"

def get_transcription_config() -> Dict[str, Any]:
    """
    Get complete transcription configuration
    
    Returns:
        dict: Configuration dictionary with all transcription settings
    """
    method = get_transcription_method()
    
    config = {
        "method": method,
        "using_deepgram": method == "deepgram",
        "using_gpu": method != "deepgram",
        "temporary_override": USE_DEEPGRAM_TEMPORARILY or os.environ.get("FORCE_DEEPGRAM_TRANSCRIPTION", "").lower() == "true",
        "environment_override": "FORCE_DEEPGRAM_TRANSCRIPTION" in os.environ,
        "config_source": "environment" if "FORCE_DEEPGRAM_TRANSCRIPTION" in os.environ else "module_constant"
    }
    
    return config

def print_transcription_status():
    """Print current transcription configuration status"""
    config = get_transcription_config()
    
    print("🎤 TRANSCRIPTION CONFIGURATION STATUS")
    print("=" * 50)
    print(f"Current method: {config['method']}")
    print(f"Using Deepgram: {config['using_deepgram']}")
    print(f"Using GPU: {config['using_gpu']}")
    print(f"Temporary override: {config['temporary_override']}")
    print(f"Environment override: {config['environment_override']}")
    print(f"Config source: {config['config_source']}")
    
    if config['using_deepgram']:
        print("\n💡 TO RE-ENABLE GPU TRANSCRIPTION:")
        print("   Option 1: Set environment variable FORCE_DEEPGRAM_TRANSCRIPTION=false")
        print("   Option 2: Change USE_DEEPGRAM_TEMPORARILY = False in transcription_config.py")
        print("   Option 3: Call set_transcription_method(False)")
    else:
        print("\n💡 TO USE DEEPGRAM TEMPORARILY:")
        print("   Option 1: Set environment variable FORCE_DEEPGRAM_TRANSCRIPTION=true")
        print("   Option 2: Change USE_DEEPGRAM_TEMPORARILY = True in transcription_config.py")
        print("   Option 3: Call set_transcription_method(True)")
    
    print("=" * 50)

# =============================================================================
# EASY SWITCHING FUNCTIONS
# =============================================================================

def enable_deepgram():
    """Enable Deepgram transcription (temporary)"""
    set_transcription_method(True)
    print("✅ Switched to Deepgram transcription")

def enable_gpu_transcription():
    """Enable GPU transcription (original setup)"""
    set_transcription_method(False)
    print("✅ Switched to GPU transcription")

def toggle_transcription_method():
    """Toggle between Deepgram and GPU transcription"""
    current_deepgram = is_using_deepgram()
    set_transcription_method(not current_deepgram)
    
    if current_deepgram:
        print("✅ Switched from Deepgram to GPU transcription")
    else:
        print("✅ Switched from GPU to Deepgram transcription")

# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def apply_transcription_config_to_environment():
    """
    Apply the current transcription configuration to environment variables
    This is called by main.py to ensure consistency
    """
    method = get_transcription_method()
    
    # Set the main transcription method
    os.environ["TRANSCRIPTION_METHOD"] = method
    
    # Set the force flag for consistency
    os.environ["FORCE_DEEPGRAM_TRANSCRIPTION"] = "true" if method == "deepgram" else "false"
    
    # Disable GPU transcription if using Deepgram
    if method == "deepgram":
        os.environ["ENABLE_GPU_TRANSCRIPTION"] = "false"
    else:
        # Don't override if already set
        if "ENABLE_GPU_TRANSCRIPTION" not in os.environ:
            os.environ["ENABLE_GPU_TRANSCRIPTION"] = "true"

def get_transcription_status_for_logging() -> str:
    """
    Get a formatted status string for logging
    
    Returns:
        str: Status string for logging
    """
    config = get_transcription_config()
    
    if config['using_deepgram']:
        if config['temporary_override']:
            return f"{config['method']} (TEMPORARY - GPU transcription disabled)"
        else:
            return config['method']
    else:
        return f"{config['method']} (GPU transcription enabled)"

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ["status", "show", "current"]:
            print_transcription_status()
        elif command in ["deepgram", "dg"]:
            enable_deepgram()
            print_transcription_status()
        elif command in ["gpu", "parakeet"]:
            enable_gpu_transcription()
            print_transcription_status()
        elif command in ["toggle", "switch"]:
            toggle_transcription_method()
            print_transcription_status()
        else:
            print("Usage: python transcription_config.py [status|deepgram|gpu|toggle]")
            print("  status   - Show current configuration")
            print("  deepgram - Switch to Deepgram transcription")
            print("  gpu      - Switch to GPU transcription")
            print("  toggle   - Toggle between methods")
    else:
        print_transcription_status()
