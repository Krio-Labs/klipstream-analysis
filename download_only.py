import asyncio
import logging
import os
import platform
import sys
from pathlib import Path
from audio_downloader import TwitchVideoDownloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def download_twitch_video(url):
    """Download a Twitch video and convert it to audio"""
    try:
        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Set up symbolic link for ffmpeg on macOS
        if platform.system() == "Darwin":
            if os.path.exists("ffmpeg_mac") and not os.path.exists("ffmpeg"):
                os.symlink("ffmpeg_mac", "ffmpeg")
                logging.info("Created symbolic link from ffmpeg_mac to ffmpeg")
        
        # Set up .NET environment for macOS
        if platform.system() == "Darwin":
            cache_dir = os.path.expanduser("~/.cache/twitch_downloader")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["DOTNET_BUNDLE_EXTRACT_BASE_DIR"] = cache_dir
            logging.info(f"Set DOTNET_BUNDLE_EXTRACT_BASE_DIR to {cache_dir}")
        
        # Initialize downloader
        downloader = TwitchVideoDownloader()
        
        # Process video
        audio_file = await downloader.process_video(url)
        
        if audio_file:
            logging.info(f"Successfully downloaded and converted video to audio: {audio_file}")
            print(f"\nAudio file saved to: {audio_file}")
            return True
        else:
            logging.error("Failed to download and convert video")
            return False
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_only.py <twitch_video_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    asyncio.run(download_twitch_video(url))
