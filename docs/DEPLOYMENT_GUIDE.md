# 🚀 KlipStream Analysis Deployment Guide

## Overview

This guide covers deploying the KlipStream Analysis service to Google Cloud Run, including setup, configuration, and maintenance procedures.

## Prerequisites

### Required Tools
- Google Cloud SDK (`gcloud`)
- Docker Desktop
- Git
- Python 3.10+

### Required Accounts
- Google Cloud Platform account with billing enabled
- Deepgram API account
- Nebius API account (for sentiment analysis)
- Convex account (for database)

## Initial Setup

### 1. Google Cloud Project Setup

```bash
# Create a new project (optional)
gcloud projects create klipstream --name="KlipStream"

# Set the project
gcloud config set project klipstream

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable storage.googleapis.com
```

### 2. Service Account Creation

```bash
# Create service account
gcloud iam service-accounts create klipstream-service \
    --display-name="KlipStream Service Account" \
    --description="Service account for KlipStream Analysis"

# Grant necessary permissions
gcloud projects add-iam-policy-binding klipstream \
    --member="serviceAccount:klipstream-service@klipstream.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding klipstream \
    --member="serviceAccount:klipstream-service@klipstream.iam.gserviceaccount.com" \
    --role="roles/run.invoker"

# Create and download service account key
gcloud iam service-accounts keys create new-service-account-key.json \
    --iam-account=klipstream-service@klipstream.iam.gserviceaccount.com
```

### 3. Google Cloud Storage Setup

```bash
# Create storage buckets
gsutil mb gs://klipstream-vods-raw
gsutil mb gs://klipstream-transcripts
gsutil mb gs://klipstream-chatlogs
gsutil mb gs://klipstream-analysis

# Set bucket permissions (public read for generated files)
gsutil iam ch allUsers:objectViewer gs://klipstream-vods-raw
gsutil iam ch allUsers:objectViewer gs://klipstream-transcripts
gsutil iam ch allUsers:objectViewer gs://klipstream-chatlogs
gsutil iam ch allUsers:objectViewer gs://klipstream-analysis
```

## Configuration

### 1. Environment Variables

Create `.env.yaml` in the project root:

```yaml
# Base configuration
BASE_DIR: "/tmp/output"
USE_GCS: "true"
GCS_PROJECT: "klipstream"

# API Keys
DEEPGRAM_API_KEY: "your_deepgram_api_key_here"
NEBIUS_API_KEY: "your_nebius_api_key_here"

# Convex Database
CONVEX_URL: "https://your-convex-deployment.convex.cloud"
CONVEX_API_KEY: "your_convex_api_key_here"
CONVEX_DEPLOYMENT: "your_deployment_name"

# Auth0 (if needed)
AUTH0_DOMAIN: "your-auth0-domain.auth0.com"
AUTH0_CLIENT_ID: "your_auth0_client_id"
AUTH0_CLIENT_SECRET: "your_auth0_client_secret"
```

### 2. Service Account Key

Ensure `new-service-account-key.json` is in the project root directory.

## Deployment Process

### 1. Automated Deployment

Use the provided deployment script:

```bash
# Make script executable
chmod +x deploy_cloud_run_simple.sh

# Run deployment
./deploy_cloud_run_simple.sh
```

### 2. Manual Deployment

If you prefer manual deployment:

```bash
# Set project
gcloud config set project klipstream

# Configure Docker
gcloud auth configure-docker

# Build image
docker buildx build --platform linux/amd64 -t gcr.io/klipstream/klipstream-analysis .

# Push image
docker push gcr.io/klipstream/klipstream-analysis

# Deploy to Cloud Run
gcloud run deploy klipstream-analysis \
  --image gcr.io/klipstream/klipstream-analysis \
  --platform managed \
  --region us-central1 \
  --project klipstream \
  --update-env-vars="BASE_DIR=/tmp/output,USE_GCS=true,GCS_PROJECT=klipstream,DEEPGRAM_API_KEY=your_key,NEBIUS_API_KEY=your_key,CONVEX_URL=your_url,CONVEX_API_KEY=your_key,CONVEX_DEPLOYMENT=your_deployment" \
  --cpu 8 \
  --memory 32Gi \
  --timeout 3600s \
  --service-account klipstream-service@klipstream.iam.gserviceaccount.com \
  --allow-unauthenticated
```

## Configuration Options

### Resource Allocation

| Setting | Value | Reason |
|---------|-------|--------|
| CPU | 8 cores | Video processing is CPU-intensive |
| Memory | 32GB | Large video files require significant RAM |
| Timeout | 3600s | Maximum Cloud Run timeout (1 hour) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BASE_DIR` | Yes | Base directory for file operations |
| `USE_GCS` | Yes | Enable Google Cloud Storage |
| `GCS_PROJECT` | Yes | Google Cloud project ID |
| `DEEPGRAM_API_KEY` | Yes | Deepgram transcription API key |
| `NEBIUS_API_KEY` | Yes | Nebius sentiment analysis API key |
| `CONVEX_URL` | Yes | Convex database URL |
| `CONVEX_API_KEY` | Yes | Convex API key |
| `CONVEX_DEPLOYMENT` | Yes | Convex deployment name |

## Monitoring and Maintenance

### 1. Health Checks

```bash
# Check service status
gcloud run services describe klipstream-analysis \
  --region us-central1 \
  --project klipstream

# Test API endpoint
curl -X POST https://your-service-url.run.app \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.twitch.tv/videos/test_video_id"}'
```

### 2. Log Monitoring

```bash
# View recent logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=klipstream-analysis" \
  --limit 50 \
  --project klipstream

# Stream logs in real-time
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=klipstream-analysis" \
  --project klipstream
```

### 3. Performance Monitoring

Monitor these metrics in Google Cloud Console:
- **Request Count**: Number of API calls
- **Request Latency**: Response time (should be < 10s for API response)
- **Error Rate**: Failed requests percentage
- **CPU Utilization**: Should stay under 80%
- **Memory Utilization**: Monitor for memory leaks

## Scaling Configuration

### Automatic Scaling

Cloud Run automatically scales based on:
- **Min Instances**: 0 (cost-effective)
- **Max Instances**: 10 (adjust based on expected load)
- **Concurrency**: 1 (one request per instance due to resource intensity)

```bash
# Update scaling settings
gcloud run services update klipstream-analysis \
  --region us-central1 \
  --project klipstream \
  --min-instances 0 \
  --max-instances 10 \
  --concurrency 1
```

## Security Configuration

### 1. Service Account Permissions

Minimal required permissions:
- `roles/storage.objectAdmin` - For GCS file operations
- `roles/run.invoker` - For service invocation

### 2. Network Security

```bash
# Restrict access to specific IPs (optional)
gcloud run services update klipstream-analysis \
  --region us-central1 \
  --project klipstream \
  --ingress internal-and-cloud-load-balancing
```

### 3. API Key Management

- Store API keys in Google Secret Manager (recommended)
- Rotate keys regularly
- Monitor API key usage

## Troubleshooting

### Common Issues

1. **Deployment Failures**
   ```bash
   # Check build logs
   gcloud builds list --project klipstream
   gcloud builds log BUILD_ID --project klipstream
   ```

2. **Service Not Responding**
   ```bash
   # Check service status
   gcloud run services describe klipstream-analysis \
     --region us-central1 \
     --project klipstream \
     --format="value(status.conditions[0].message)"
   ```

3. **Memory Issues**
   ```bash
   # Increase memory allocation
   gcloud run services update klipstream-analysis \
     --region us-central1 \
     --project klipstream \
     --memory 64Gi
   ```

4. **Timeout Issues**
   - Verify timeout is set to maximum (3600s)
   - Check if video files are exceptionally large
   - Monitor processing logs for bottlenecks

### Log Analysis

Common log patterns to monitor:
- `ERROR` - Processing failures
- `TIMEOUT` - Request timeouts
- `Memory` - Memory-related issues
- `Storage` - GCS upload/download issues

## Backup and Recovery

### 1. Configuration Backup

```bash
# Export current service configuration
gcloud run services describe klipstream-analysis \
  --region us-central1 \
  --project klipstream \
  --format export > service-config-backup.yaml
```

### 2. Data Backup

- GCS buckets are automatically replicated
- Convex database has built-in backups
- Service account keys should be stored securely

### 3. Disaster Recovery

1. **Service Recovery**:
   ```bash
   # Redeploy from backup configuration
   gcloud run services replace service-config-backup.yaml \
     --region us-central1 \
     --project klipstream
   ```

2. **Data Recovery**:
   - GCS: Use versioning and lifecycle policies
   - Convex: Contact support for data recovery

## Cost Optimization

### 1. Resource Optimization

- **CPU**: Start with 4 cores, scale up if needed
- **Memory**: Monitor usage and adjust accordingly
- **Timeout**: Use minimum required timeout
- **Min Instances**: Keep at 0 for cost savings

### 2. Storage Optimization

```bash
# Set lifecycle policies for old files
gsutil lifecycle set lifecycle-config.json gs://klipstream-vods-raw
```

Example `lifecycle-config.json`:
```json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }
  ]
}
```

### 3. Monitoring Costs

- Set up billing alerts
- Monitor Cloud Run usage in billing console
- Review GCS storage costs monthly

## Updates and Maintenance

### 1. Regular Updates

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Rebuild and redeploy
./deploy_cloud_run_simple.sh
```

### 2. Security Updates

- Update base Docker image regularly
- Keep Python dependencies updated
- Monitor security advisories

### 3. Performance Tuning

- Monitor processing times
- Optimize code based on logs
- Adjust resource allocation as needed

This deployment guide ensures a robust, scalable, and maintainable KlipStream Analysis service on Google Cloud Run.
