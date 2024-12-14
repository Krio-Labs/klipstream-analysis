# Twitch Video Downloader with Audio Level Analysis

This Google Cloud Function downloads the lowest-quality rendition of a Twitch video using the **TwitchDownloaderCLI**, extracts the audio using **FFmpeg**, and uploads the audio file to Ci.

## Features

- Downloads the lowest quality (160p) Twitch video.
- Extracts audio from the video using FFmpeg.
- Uploads the audio file to Ci.

## Requirements

- Google Cloud Project with Cloud Functions enabled.
- Google Cloud SDK installed locally.
- Proper IAM permissions (to deploy and access Cloud Functions).
- TwitchDownloaderCLI and FFmpeg binaries are included in the function.

## Setup

### 1. Prerequisites

Ensure the following tools are installed and configured locally:

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [Python 3.11](https://www.python.org/downloads/)
- [TwitchDownloaderCLI](https://github.com/lay295/TwitchDownloader) (included in the function)
- [FFmpeg](https://ffmpeg.org/) (included in the function)

### 2. Project Structure

twitch-video-downloader/
│
├── ffmpeg                     # FFmpeg binary
├── TwitchDownloaderCLI        # TwitchDownloaderCLI binary
├── main.py                    # Python Cloud Function code
├── requirements.txt           # Empty or with additional dependencies (if needed)

### 3. Deploy the Cloud Function

For the function to work, it needs to upload the required binaries which are ignored in the .gitignore file. 

To deploy the function, you need to download the binaries and un-ignore them in the .gitignore file.

```gitignore
# TwitchDownloaderCLI
# TwitchDownloaderCLI.exe
# ffmpeg
# ffmpeg.exe
```

After that, you can deploy the function using the following command:

```bash
gcloud functions deploy Chat-Audio-Analytics --gen2 --runtime python312 --region us-central1 --source . --trigger-http --allow-unauthenticated --entry-point run_pipeline --timeout 3600s --memory 2GB
```
