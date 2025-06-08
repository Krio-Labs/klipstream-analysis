#!/bin/bash

# GPU-Enabled Cloud Run Deployment Script for KlipStream Analysis
# This script deploys the klipstream-analysis service with NVIDIA L4 GPU support

set -e

# Configuration
PROJECT_ID="klipstream"
REGION="us-central1"
SERVICE_NAME="klipstream-analysis"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# GPU Configuration
GPU_TYPE="nvidia-l4"
GPU_COUNT="1"
CPU="8"
MEMORY="32Gi"
TIMEOUT="3600"

# Service Account
SERVICE_ACCOUNT_EMAIL="klipstream-analysis@${PROJECT_ID}.iam.gserviceaccount.com"

# Environment Variables for GPU Transcription
ENV_VARS="ENABLE_GPU_TRANSCRIPTION=true"
ENV_VARS="${ENV_VARS},TRANSCRIPTION_METHOD=auto"
ENV_VARS="${ENV_VARS},PARAKEET_MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2"
ENV_VARS="${ENV_VARS},GPU_BATCH_SIZE=8"
ENV_VARS="${ENV_VARS},GPU_MEMORY_LIMIT_GB=20"
ENV_VARS="${ENV_VARS},CHUNK_DURATION_MINUTES=10"
ENV_VARS="${ENV_VARS},ENABLE_BATCH_PROCESSING=true"
ENV_VARS="${ENV_VARS},ENABLE_FALLBACK=true"
ENV_VARS="${ENV_VARS},COST_OPTIMIZATION=true"
ENV_VARS="${ENV_VARS},SHORT_FILE_THRESHOLD_HOURS=2"
ENV_VARS="${ENV_VARS},LONG_FILE_THRESHOLD_HOURS=4"
ENV_VARS="${ENV_VARS},ENABLE_PERFORMANCE_METRICS=true"
ENV_VARS="${ENV_VARS},LOG_TRANSCRIPTION_COSTS=true"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 GPU-Enabled Cloud Run Deployment${NC}"
echo "=================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo "GPU: ${GPU_COUNT}x ${GPU_TYPE}"
echo "CPU: ${CPU}"
echo "Memory: ${MEMORY}"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Check if gcloud is authenticated
echo -e "${BLUE}🔐 Checking authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${RED}❌ Not authenticated with gcloud. Please run 'gcloud auth login'${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Authenticated${NC}"

# Set project
echo -e "${BLUE}📋 Setting project...${NC}"
gcloud config set project ${PROJECT_ID}
echo -e "${GREEN}✅ Project set to ${PROJECT_ID}${NC}"

# Enable required APIs
echo -e "${BLUE}🔧 Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
echo -e "${GREEN}✅ APIs enabled${NC}"

# Build Docker image with GPU support
echo -e "${BLUE}🐳 Building GPU-enabled Docker image...${NC}"
if [ -f "Dockerfile.gpu" ]; then
    echo "Using Dockerfile.gpu for GPU build..."
    gcloud builds submit --tag ${IMAGE_NAME} --file Dockerfile.gpu .
else
    echo -e "${YELLOW}⚠️  Dockerfile.gpu not found, using standard Dockerfile${NC}"
    echo "Note: Make sure your Dockerfile includes GPU support"
    gcloud builds submit --tag ${IMAGE_NAME} .
fi
echo -e "${GREEN}✅ Docker image built and pushed${NC}"

# Check GPU quota
echo -e "${BLUE}🎯 Checking GPU quota...${NC}"
QUOTA_CHECK=$(gcloud compute project-info describe --format="value(quotas[].limit)" --filter="quotas.metric:NVIDIA_L4_GPUS" 2>/dev/null || echo "0")
if [ "$QUOTA_CHECK" = "0" ] || [ -z "$QUOTA_CHECK" ]; then
    echo -e "${YELLOW}⚠️  Warning: NVIDIA L4 GPU quota may not be available${NC}"
    echo "You may need to request GPU quota increase:"
    echo "https://console.cloud.google.com/iam-admin/quotas"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 1
    fi
else
    echo -e "${GREEN}✅ GPU quota available: ${QUOTA_CHECK}${NC}"
fi

# Deploy to Cloud Run with GPU
echo -e "${BLUE}🚀 Deploying to Cloud Run with GPU...${NC}"
echo "This may take several minutes..."

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --gpu=${GPU_COUNT} \
    --gpu-type=${GPU_TYPE} \
    --cpu ${CPU} \
    --memory ${MEMORY} \
    --timeout ${TIMEOUT}s \
    --service-account ${SERVICE_ACCOUNT_EMAIL} \
    --allow-unauthenticated \
    --max-instances 5 \
    --min-instances 0 \
    --concurrency 1 \
    --port 8080 \
    --set-env-vars="${ENV_VARS}" \
    --execution-environment gen2

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
else
    echo -e "${RED}❌ Deployment failed${NC}"
    exit 1
fi

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")
echo ""
echo -e "${GREEN}🎉 GPU-enabled service deployed successfully!${NC}"
echo "Service URL: ${SERVICE_URL}"
echo ""

# Test GPU availability
echo -e "${BLUE}🧪 Testing GPU availability...${NC}"
echo "Testing endpoint: ${SERVICE_URL}/health"

# Wait a moment for service to be ready
sleep 10

HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health" || echo "000")
if [ "$HEALTH_CHECK" = "200" ]; then
    echo -e "${GREEN}✅ Service health check passed${NC}"
else
    echo -e "${YELLOW}⚠️  Health check returned: ${HEALTH_CHECK}${NC}"
    echo "Service may still be starting up..."
fi

# Display configuration summary
echo ""
echo -e "${BLUE}📋 Deployment Summary${NC}"
echo "===================="
echo "Service: ${SERVICE_NAME}"
echo "URL: ${SERVICE_URL}"
echo "GPU: ${GPU_COUNT}x ${GPU_TYPE}"
echo "Resources: ${CPU} CPU, ${MEMORY} memory"
echo "Timeout: ${TIMEOUT} seconds"
echo "Max Instances: 5"
echo "Concurrency: 1 (GPU exclusive)"
echo ""

echo -e "${BLUE}🔧 GPU Transcription Configuration${NC}"
echo "================================="
echo "GPU Transcription: Enabled"
echo "Method Selection: Auto"
echo "Model: nvidia/parakeet-tdt-0.6b-v2"
echo "Batch Size: 8"
echo "Chunk Duration: 10 minutes"
echo "Fallback: Enabled"
echo "Cost Optimization: Enabled"
echo ""

echo -e "${BLUE}📊 Monitoring & Logs${NC}"
echo "=================="
echo "View logs:"
echo "gcloud run services logs tail ${SERVICE_NAME} --region=${REGION}"
echo ""
echo "Monitor service:"
echo "https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}"
echo ""

echo -e "${BLUE}🧪 Testing Commands${NC}"
echo "=================="
echo "Test transcription endpoint:"
echo "curl -X POST \"${SERVICE_URL}/transcribe\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"video_url\": \"YOUR_VIDEO_URL\"}'"
echo ""

echo -e "${BLUE}💰 Cost Monitoring${NC}"
echo "=================="
echo "Expected costs for GPU usage:"
echo "- NVIDIA L4: ~\$0.45/hour when active"
echo "- CPU/Memory: ~\$0.10/hour when active"
echo "- Storage: ~\$0.02/GB/month"
echo ""
echo "Monitor costs at:"
echo "https://console.cloud.google.com/billing"
echo ""

echo -e "${GREEN}🎉 GPU-enabled KlipStream Analysis is ready!${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Test the service with a sample video"
echo "2. Monitor GPU utilization and costs"
echo "3. Adjust configuration as needed"
echo "4. Set up monitoring and alerting"
echo ""

# Save deployment info
cat > deployment_info.json << EOF
{
  "deployment_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "service_name": "${SERVICE_NAME}",
  "service_url": "${SERVICE_URL}",
  "project_id": "${PROJECT_ID}",
  "region": "${REGION}",
  "gpu_config": {
    "type": "${GPU_TYPE}",
    "count": ${GPU_COUNT}
  },
  "resources": {
    "cpu": "${CPU}",
    "memory": "${MEMORY}",
    "timeout": ${TIMEOUT}
  },
  "gpu_transcription_enabled": true,
  "environment_variables": "${ENV_VARS}"
}
EOF

echo -e "${BLUE}💾 Deployment info saved to deployment_info.json${NC}"
