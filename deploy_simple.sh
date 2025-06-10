#!/bin/bash

# Simple Cloud Run Deployment Script
# Deploys CPU-only version for quick testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="klipstream"
REGION="us-central1"
SERVICE_NAME="klipstream-analysis"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
DOCKERFILE="Dockerfile.simple"

echo -e "${BLUE}🚀 Simple Cloud Run Deployment${NC}"
echo "=================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo "CPU: 4"
echo "Memory: 16Gi"
echo "Timeout: 3600s"
echo "Mode: CPU-only (Deepgram transcription)"
echo ""

# Check authentication
echo -e "${BLUE}🔐 Checking authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${RED}❌ Not authenticated with gcloud${NC}"
    echo "Please run: gcloud auth login"
    exit 1
fi
echo -e "${GREEN}✅ Authenticated${NC}"

# Set project
echo -e "${BLUE}📋 Setting project...${NC}"
gcloud config set project ${PROJECT_ID}
echo -e "${GREEN}✅ Project set to ${PROJECT_ID}${NC}"

# Enable required APIs
echo -e "${BLUE}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
echo -e "${GREEN}✅ APIs enabled${NC}"

# Environment Variables for CPU Transcription
ENV_VARS="ENABLE_GPU_TRANSCRIPTION=false"
ENV_VARS="${ENV_VARS},TRANSCRIPTION_METHOD=deepgram"
ENV_VARS="${ENV_VARS},ENABLE_FALLBACK=true"
ENV_VARS="${ENV_VARS},COST_OPTIMIZATION=true"

# API Configuration
ENV_VARS="${ENV_VARS},FASTAPI_MODE=true"
ENV_VARS="${ENV_VARS},API_VERSION=2.0.0"
ENV_VARS="${ENV_VARS},ENABLE_ASYNC_API=true"

# Cloud Environment Configuration
ENV_VARS="${ENV_VARS},CLOUD_ENVIRONMENT=true"
ENV_VARS="${ENV_VARS},USE_CLOUD_STORAGE=true"
ENV_VARS="${ENV_VARS},GCS_PROJECT_ID=klipstream"

# Convex Configuration
ENV_VARS="${ENV_VARS},CONVEX_URL=${CONVEX_URL:-}"

# Deepgram Configuration
ENV_VARS="${ENV_VARS},DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY:-}"

# Nebius Configuration (for sentiment analysis)
ENV_VARS="${ENV_VARS},NEBIUS_API_KEY=${NEBIUS_API_KEY:-}"

# Build and deploy
echo -e "${BLUE}🐳 Building simple Docker image...${NC}"
echo "Using ${DOCKERFILE} for CPU build..."

# Copy Dockerfile.simple to Dockerfile for build
cp ${DOCKERFILE} Dockerfile

gcloud builds submit --tag ${IMAGE_NAME} .

# Restore original Dockerfile
if [ -f Dockerfile.gpu ]; then
    cp Dockerfile.gpu Dockerfile
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Deploy to Cloud Run
echo -e "${BLUE}🚀 Deploying to Cloud Run...${NC}"

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 16Gi \
    --cpu 4 \
    --timeout 3600 \
    --concurrency 5 \
    --max-instances 3 \
    --set-env-vars "${ENV_VARS}"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Cloud Run deployment failed${NC}"
    exit 1
fi

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

echo ""
echo -e "${GREEN}🎉 Deployment successful!${NC}"
echo "=================================="
echo -e "${GREEN}Service URL: ${SERVICE_URL}${NC}"
echo ""

echo -e "${BLUE}🧪 Testing Commands${NC}"
echo "=================="
echo "Test API health:"
echo "curl \"${SERVICE_URL}/health\""
echo ""
echo "Test API info:"
echo "curl \"${SERVICE_URL}/\""
echo ""
echo "Test transcription methods:"
echo "curl \"${SERVICE_URL}/api/v1/transcription/methods\""
echo ""
echo "Start video analysis:"
echo "curl -X POST \"${SERVICE_URL}/api/v1/analysis\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"url\": \"https://www.twitch.tv/videos/YOUR_VIDEO_ID\"}'"
echo ""

echo -e "${GREEN}✅ Simple deployment completed!${NC}"
echo -e "${YELLOW}📝 Note: This is a CPU-only deployment using Deepgram for transcription${NC}"
