#!/usr/bin/env python3
"""
Create Test Audio Files

This script helps create or find long audio files for testing the Parakeet transcriber.
It can either generate synthetic audio or help you download public domain audio.

Usage:
    python create_test_audio.py --duration 3 --output test_3hour_audio.wav
    python create_test_audio.py --download-librivox --duration 3
"""

import argparse
import sys
import os
from pathlib import Path
import requests
from pydub import AudioSegment
from pydub.generators import Sine, Sawtooth
import random

def generate_synthetic_audio(duration_hours: float, output_path: str):
    """Generate synthetic audio with speech-like patterns"""
    
    print(f"🎵 Generating {duration_hours} hours of synthetic audio...")
    
    # Create base tone
    duration_ms = int(duration_hours * 3600 * 1000)  # Convert to milliseconds
    
    # Generate segments with varying tones to simulate speech patterns
    segments = []
    segment_length = 5000  # 5 seconds per segment
    
    for i in range(0, duration_ms, segment_length):
        # Random frequency between 100-400 Hz (speech range)
        freq = random.randint(100, 400)
        
        # Create tone segment
        if random.choice([True, False]):
            tone = Sine(freq).to_audio_segment(duration=min(segment_length, duration_ms - i))
        else:
            tone = Sawtooth(freq).to_audio_segment(duration=min(segment_length, duration_ms - i))
        
        # Add some silence between "words"
        silence_duration = random.randint(100, 500)
        silence = AudioSegment.silent(duration=silence_duration)
        
        # Reduce volume to be more pleasant
        tone = tone - 20  # Reduce by 20dB
        
        segments.append(tone + silence)
        
        if i % (60 * 1000) == 0:  # Progress every minute
            print(f"  Generated {i // (60 * 1000)} minutes...")
    
    # Combine all segments
    print("🔗 Combining audio segments...")
    final_audio = sum(segments)
    
    # Export
    print(f"💾 Saving to {output_path}...")
    final_audio.export(output_path, format="wav")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Created {duration_hours} hour audio file: {output_path} ({file_size:.1f} MB)")

def download_librivox_audio(duration_hours: float, output_path: str):
    """Download public domain audiobooks from LibriVox"""
    
    print(f"📚 Downloading LibriVox audiobook content...")
    
    # Some sample LibriVox URLs (public domain)
    sample_urls = [
        "https://archive.org/download/alice_in_wonderland_librivox/alice_in_wonderland_01_carroll.mp3",
        "https://archive.org/download/pride_prejudice_librivox/pride_prejudice_01_austen.mp3",
        "https://archive.org/download/sherlock_holmes_librivox/sherlock_holmes_01_doyle.mp3"
    ]
    
    downloaded_segments = []
    total_duration = 0
    target_duration = duration_hours * 3600  # seconds
    
    for i, url in enumerate(sample_urls):
        if total_duration >= target_duration:
            break
            
        try:
            print(f"📥 Downloading segment {i+1}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            temp_file = f"temp_segment_{i}.mp3"
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Load and check duration
            audio = AudioSegment.from_mp3(temp_file)
            segment_duration = len(audio) / 1000  # seconds
            
            downloaded_segments.append(audio)
            total_duration += segment_duration
            
            print(f"  ✅ Downloaded {segment_duration/60:.1f} minutes")
            
            # Clean up temp file
            os.remove(temp_file)
            
        except Exception as e:
            print(f"  ❌ Failed to download {url}: {e}")
            continue
    
    if not downloaded_segments:
        print("❌ No audio segments downloaded successfully")
        return False
    
    # Repeat segments to reach target duration
    print("🔄 Repeating segments to reach target duration...")
    final_audio = AudioSegment.empty()
    
    while len(final_audio) / 1000 < target_duration:
        for segment in downloaded_segments:
            final_audio += segment
            if len(final_audio) / 1000 >= target_duration:
                break
    
    # Trim to exact duration
    final_audio = final_audio[:int(target_duration * 1000)]
    
    # Export
    print(f"💾 Saving to {output_path}...")
    final_audio.export(output_path, format="wav")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    actual_duration = len(final_audio) / 1000 / 3600
    print(f"✅ Created {actual_duration:.2f} hour audio file: {output_path} ({file_size:.1f} MB)")
    return True

def suggest_existing_files():
    """Suggest ways to find existing long audio files"""
    
    print("\n📁 FINDING EXISTING LONG AUDIO FILES:")
    print("\n1. 🎵 Music/Podcast Collections:")
    print("   - Check ~/Music/ for long music files or podcasts")
    print("   - Look for audiobooks in your library")
    print("   - Search for .mp3, .wav, .m4a files > 100MB")
    
    print("\n2. 🎙️ Recording Options:")
    print("   - Record a long meeting or lecture")
    print("   - Use system audio recording tools")
    print("   - Record streaming content (with permission)")
    
    print("\n3. 📚 Free Resources:")
    print("   - LibriVox.org (public domain audiobooks)")
    print("   - Archive.org audio collections")
    print("   - Podcast archives")
    
    print("\n4. 🔍 Search Commands:")
    print("   find ~/Music -name '*.mp3' -size +100M")
    print("   find ~/Downloads -name '*.m4a' -size +50M")
    print("   find / -name '*.wav' -size +200M 2>/dev/null")

def main():
    parser = argparse.ArgumentParser(description="Create or find long audio files for testing")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in hours (default: 3)")
    parser.add_argument("--output", default="test_long_audio.wav", help="Output file path")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic audio")
    parser.add_argument("--download-librivox", action="store_true", help="Download from LibriVox")
    parser.add_argument("--suggest-existing", action="store_true", help="Show suggestions for finding existing files")
    
    args = parser.parse_args()
    
    if args.suggest_existing:
        suggest_existing_files()
        return
    
    if args.duration > 5:
        print(f"⚠️  WARNING: {args.duration} hours is a very long audio file.")
        print("This will create a large file and take significant time to process.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    try:
        if args.synthetic:
            generate_synthetic_audio(args.duration, args.output)
        elif args.download_librivox:
            if not download_librivox_audio(args.duration, args.output):
                print("❌ Failed to download LibriVox content")
                sys.exit(1)
        else:
            print("Please specify --synthetic or --download-librivox or --suggest-existing")
            suggest_existing_files()
            
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
