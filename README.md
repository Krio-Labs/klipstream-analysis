# Twitch Analysis Cloud Run Service

This Google Cloud Run service analyzes Twitch VODs and uploads the results to Convex. It's designed to work with the KlipStream frontend project.

## Features

- Downloads Twitch VODs using TwitchDownloaderCLI
- Extracts audio and converts it to WAV format
- Transcribes the audio using AssemblyAI
- Performs sentiment analysis on the transcription
- Analyzes chat data and sentiment
- Identifies highlights and emotional patterns
- Uploads all results to Convex storage

## Integration with Frontend

This service is designed to work with the KlipStream frontend project. The workflow is:

1. The frontend creates a video entry in the Convex database when a user submits a Twitch VOD URL
2. The frontend then calls this Cloud Run service with the Twitch VOD URL
3. This service processes the video and updates the existing Convex record with analysis results using the `convex_upload.py` module

## Requirements

- Google Cloud Project with Cloud Run enabled
- Docker installed locally
- Google Cloud SDK installed locally
- AssemblyAI API key
- Google API key (for sentiment analysis)
- Convex account and project

## Setup

### 1. Configure Environment Variables

Update the `.env.yaml` file with your API keys:

```yaml
ASSEMBLYAI_API_KEY: "your_assemblyai_api_key_here"
GOOGLE_API_KEY: "your_google_api_key_here"
CONVEX_UPLOAD_URL: "your_convex_upload_endpoint_here"
```

### 2. Update the Existing Cloud Run Service

Run the update script to update the existing "Chat-Audio-Analytics" service:

```bash
./update_cloud_run.sh
```

For more detailed instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

### 3. Manual Deployment (Alternative)

If you prefer to deploy manually:

```bash
# Build the Docker image
docker build -t gcr.io/optimum-habitat-429714-a7/Chat-Audio-Analytics .

# Push the image to Google Container Registry
docker push gcr.io/optimum-habitat-429714-a7/Chat-Audio-Analytics

# Update the Cloud Run service
gcloud run deploy Chat-Audio-Analytics \
    --image gcr.io/optimum-habitat-429714-a7/Chat-Audio-Analytics \
    --platform managed \
    --region us-central1 \
    --project optimum-habitat-429714-a7 \
    --env-vars-file .env.yaml \
    --allow-unauthenticated
```

## Usage

Send a POST request to the Cloud Run service URL:

```json
{
  "url": "https://www.twitch.tv/videos/YOUR_VIDEO_ID"
}
```

**Important**: The Twitch video must already exist in your Convex database before calling this service.

## Testing

To test locally:

```bash
# Install the Functions Framework
pip install functions-framework

# Run the function locally
functions-framework --target=run_pipeline --debug

# In another terminal, use the test script
python test_cloud_function.py
```
