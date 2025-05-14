# Deploying KlipStream Analysis

This document provides detailed instructions for deploying the KlipStream Analysis pipeline to Google Cloud.

## Prerequisites

Before deploying, ensure you have:

1. **Google Cloud Project**: A Google Cloud project with billing enabled
2. **Google Cloud SDK**: Installed and configured on your local machine
3. **Docker**: Installed locally (for local builds)
4. **Service Account**: A service account with appropriate permissions
5. **API Keys**: Required API keys for Deepgram and Nebius
6. **GCS Buckets**: The following buckets created in Google Cloud Storage:
   - `klipstream-vods-raw`: For video, audio, and waveform files
   - `klipstream-transcripts`: For transcript files
   - `klipstream-chatlogs`: For chat log files
   - `klipstream-analysis`: For integrated analysis files

## Environment Configuration

Create a `.env.yaml` file in the project root with the following content:

```yaml
DEEPGRAM_API_KEY: "your_deepgram_api_key"
GOOGLE_API_KEY: "your_google_api_key"
NEBIUS_API_KEY: "your_nebius_api_key"
GCS_PROJECT: "your_gcs_project_id"
GCS_VOD_BUCKET: "klipstream-vods-raw"
GCS_TRANSCRIPT_BUCKET: "klipstream-transcripts"
GCS_CHATLOG_BUCKET: "klipstream-chatlogs"
GCS_ANALYSIS_BUCKET: "klipstream-analysis"
```

## Service Account Setup

1. Create a service account in the Google Cloud Console:
   ```bash
   gcloud iam service-accounts create klipstream-service \
       --display-name="KlipStream Service Account"
   ```

2. Grant the service account the necessary permissions:
   ```bash
   gcloud projects add-iam-policy-binding your-project-id \
       --member="serviceAccount:klipstream-service@your-project-id.iam.gserviceaccount.com" \
       --role="roles/storage.objectAdmin"
   ```

3. Create and download a service account key:
   ```bash
   gcloud iam service-accounts keys create new-service-account-key.json \
       --iam-account=klipstream-service@your-project-id.iam.gserviceaccount.com
   ```

4. Place the key file in the project root directory.

## Deployment Options

### Option 1: Cloud Run Deployment (Recommended)

Cloud Run is recommended for production use as it provides:
- Longer execution times (up to 2 hours)
- More memory (up to 32GB)
- More CPU (up to 8 vCPU cores)
- Better scaling capabilities

#### Using the Deployment Script

1. Run the deployment script:
   ```bash
   ./deploy_cloud_run.sh
   ```

2. Follow the prompts to select:
   - Deployment method (local Docker build or Cloud Build)
   - Resource configuration (CPU, memory, timeout)

3. The script will:
   - Build the Docker image
   - Push it to Google Container Registry
   - Deploy it to Cloud Run
   - Configure environment variables
   - Set up service account permissions

#### Manual Deployment Steps

If you prefer to deploy manually, follow these steps:

1. Build the Docker image:
```bash
docker build -t gcr.io/optimum-habitat-429714-a7/klipstream-analysis .
```

2. Push the image to Google Container Registry:
```bash
docker push gcr.io/optimum-habitat-429714-a7/klipstream-analysis
```

3. Deploy the Cloud Run service:
```bash
gcloud run deploy klipstream-analysis \
    --image gcr.io/optimum-habitat-429714-a7/klipstream-analysis \
    --platform managed \
    --region us-central1 \
    --project optimum-habitat-429714-a7 \
    --env-vars-file .env.yaml \
    --allow-unauthenticated \
    --service-account="klipstream-service@optimum-habitat-429714-a7.iam.gserviceaccount.com" \
    --cpu=4 \
    --memory=8Gi \
    --timeout=3600s
```

4. Grant GCS permissions to the service account:
```bash
gcloud projects add-iam-policy-binding optimum-habitat-429714-a7 \
    --member="serviceAccount:klipstream-service@optimum-habitat-429714-a7.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

### Option 2: Cloud Function Deployment

For development or testing, you can deploy as a Google Cloud Function:

```bash
./deploy_cloud_function.sh
```

This deploys a 2nd generation Cloud Function with 16GB memory and 60-minute timeout.

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
