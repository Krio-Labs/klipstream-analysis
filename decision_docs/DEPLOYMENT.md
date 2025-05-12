# Deploying the Twitch Analysis Service

This document provides instructions for deploying the Twitch Analysis service to Google Cloud Run.

## Prerequisites

1. Google Cloud Platform account with Cloud Run enabled
2. Docker installed on your local machine
3. Google Cloud CLI (gcloud) installed and configured
4. Deepgram API key
5. Google API key (for sentiment analysis)
6. Google Cloud Storage buckets created:
   - `klipstream-vods-raw`
   - `klipstream-transcripts`
   - `klipstream-chatlogs`
7. Service account with Storage Object Admin permissions

## Updating the Existing Cloud Run Service

This project is set up to update the existing "Chat-Audio-Analytics" Cloud Run service. To update the service:

1. Update the `.env.yaml` file with your API keys and GCS configuration:

```yaml
DEEPGRAM_API_KEY: "your_deepgram_api_key_here"
GOOGLE_API_KEY: "your_google_api_key_here"
GCS_PROJECT: "klipstream"
GCS_VOD_BUCKET: "klipstream-vods-raw"
GCS_TRANSCRIPT_BUCKET: "klipstream-transcripts"
GCS_CHATLOG_BUCKET: "klipstream-chatlogs"
```

2. Run the update script:

```bash
./update_cloud_run.sh
```

This script will:
- Build a Docker image with your code
- Push the image to Google Container Registry
- Update the existing Cloud Run service with the new image

## Manual Deployment Steps

If you prefer to deploy manually, follow these steps:

1. Build the Docker image:
```bash
docker build -t gcr.io/optimum-habitat-429714-a7/Chat-Audio-Analytics .
```

2. Push the image to Google Container Registry:
```bash
docker push gcr.io/optimum-habitat-429714-a7/Chat-Audio-Analytics
```

3. Update the Cloud Run service:
```bash
gcloud run deploy Chat-Audio-Analytics \
    --image gcr.io/optimum-habitat-429714-a7/Chat-Audio-Analytics \
    --platform managed \
    --region us-central1 \
    --project optimum-habitat-429714-a7 \
    --env-vars-file .env.yaml \
    --allow-unauthenticated \
    --service-account="klipstream-service@optimum-habitat-429714-a7.iam.gserviceaccount.com"
```

4. Grant GCS permissions to the service account:
```bash
gcloud projects add-iam-policy-binding optimum-habitat-429714-a7 \
    --member="serviceAccount:klipstream-service@optimum-habitat-429714-a7.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

## Testing

To test the service locally before deployment:

1. Install the Functions Framework: `pip install functions-framework`
2. Run the function locally: `functions-framework --target=run_pipeline --debug`
3. Send a test request to `http://localhost:8080`
4. Use the `test_cloud_function.py` script to test the integration

To test the GCS upload functionality specifically:

```bash
# Test uploading files to GCS
python test_gcs_upload.py --video-id YOUR_VIDEO_ID --test-upload

# Test updating video status
python test_gcs_upload.py --video-id YOUR_VIDEO_ID --test-status
```

## Troubleshooting

- If you encounter memory issues, consider increasing the memory allocation in the Cloud Run service settings
- For larger videos, you may need to increase the timeout settings
- Check the Cloud Run logs for detailed error messages
- If you get a "Video not found" error, make sure the video exists in your database first
- If you encounter GCS permission issues, verify that the service account has the correct permissions
- Make sure the GCS buckets exist and are accessible to the service account
