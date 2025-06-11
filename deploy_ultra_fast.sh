#!/bin/bash

# Ultra-fast deployment using pre-built base image
# Expected time: 1-2 minutes

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ID="klipstream"
REGION="us-central1"
SERVICE_NAME="klipstream-analysis"

echo -e "${BLUE}⚡ Ultra-Fast Cloud Run Deployment${NC}"
echo "================================="

# Build and deploy using pre-built base
# Create temporary cloudbuild config for prebuilt dockerfile
cat > cloudbuild.prebuilt.yaml << 'EOFBUILD'
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-f', 'Dockerfile.prebuilt', '-t', 'gcr.io/${PROJECT_ID}/klipstream-analysis', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/${PROJECT_ID}/klipstream-analysis']
options:
  machineType: 'E2_HIGHCPU_8'
  diskSizeGb: 50
EOFBUILD

gcloud builds submit --config=cloudbuild.prebuilt.yaml .

# Deploy to Cloud Run
gcloud run deploy ${SERVICE_NAME} \
    --image gcr.io/${PROJECT_ID}/klipstream-analysis \
    --platform managed \
    --region ${REGION} \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --timeout=3600s \
    --allow-unauthenticated \
    --max-instances=3 \
    --min-instances=0 \
    --concurrency=1 \
    --port=8080 \
    --execution-environment=gen2

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo -e "${GREEN}⚡ Ultra-fast deployment complete!${NC}"
echo "Service URL: ${SERVICE_URL}"
echo "Deployment time: ~1-2 minutes"
