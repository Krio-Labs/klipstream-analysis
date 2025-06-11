#!/bin/bash

# Enhanced GPU-Optimized Cloud Run Deployment Script
# Deploys klipstream-analysis with comprehensive GPU optimization features

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

# GPU Configuration
GPU_TYPE="nvidia-l4"
GPU_COUNT="1"
CPU="8"
MEMORY="32Gi"
TIMEOUT="3600"

echo -e "${BLUE}🚀 ENHANCED GPU-OPTIMIZED DEPLOYMENT${NC}"
echo "=" * 60
echo -e "${BLUE}Project: ${PROJECT_ID}${NC}"
echo -e "${BLUE}Service: ${SERVICE_NAME}${NC}"
echo -e "${BLUE}Region: ${REGION}${NC}"
echo -e "${BLUE}GPU: ${GPU_COUNT}x ${GPU_TYPE}${NC}"
echo -e "${BLUE}Resources: ${CPU} CPU, ${MEMORY} memory${NC}"
echo ""

# Enhanced GPU Optimization Environment Variables
echo -e "${BLUE}🔧 Configuring enhanced GPU optimization...${NC}"

# Core optimization features
ENV_VARS="ENABLE_GPU_TRANSCRIPTION=true"
ENV_VARS="${ENV_VARS},TRANSCRIPTION_METHOD=auto"
ENV_VARS="${ENV_VARS},PARAKEET_MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2"

# Enhanced GPU optimization features
ENV_VARS="${ENV_VARS},ENABLE_AMP=true"
ENV_VARS="${ENV_VARS},ENABLE_MEMORY_OPTIMIZATION=true"
ENV_VARS="${ENV_VARS},ENABLE_PARALLEL_CHUNKING=true"
ENV_VARS="${ENV_VARS},ENABLE_DEVICE_OPTIMIZATION=true"
ENV_VARS="${ENV_VARS},ENABLE_PERFORMANCE_MONITORING=true"

# AMP Configuration
ENV_VARS="${ENV_VARS},AMP_DTYPE=float16"

# Memory Optimization
ENV_VARS="${ENV_VARS},MEMORY_CLEANUP_THRESHOLD=0.8"
ENV_VARS="${ENV_VARS},MAX_MEMORY_FRAGMENTATION=0.3"
ENV_VARS="${ENV_VARS},MEMORY_MONITORING_INTERVAL=1.0"

# Parallel Processing
ENV_VARS="${ENV_VARS},MAX_CHUNK_WORKERS=8"
ENV_VARS="${ENV_VARS},CHUNK_IO_THROTTLE_DELAY=0.01"

# Device Optimizations
ENV_VARS="${ENV_VARS},CUDA_BENCHMARK=true"
ENV_VARS="${ENV_VARS},CUDA_TF32=true"

# Performance Monitoring
ENV_VARS="${ENV_VARS},SAVE_PERFORMANCE_METRICS=true"
ENV_VARS="${ENV_VARS},PERFORMANCE_METRICS_FILE=/tmp/transcription_performance_metrics.json"

# Batch Processing
ENV_VARS="${ENV_VARS},MIN_BATCH_SIZE=1"
ENV_VARS="${ENV_VARS},MAX_BATCH_SIZE=16"

# Cloud Run Specific
ENV_VARS="${ENV_VARS},CLOUD_RUN_ENVIRONMENT=true"
ENV_VARS="${ENV_VARS},CLOUD_RUN_MEMORY_GB=32"
ENV_VARS="${ENV_VARS},CLOUD_RUN_TIMEOUT_SECONDS=3600"

# Fallback Configuration
ENV_VARS="${ENV_VARS},ENABLE_GPU_FALLBACK=true"
ENV_VARS="${ENV_VARS},ENABLE_ENHANCED_FALLBACK=true"
ENV_VARS="${ENV_VARS},FALLBACK_RETRY_ATTEMPTS=3"

# Compatibility
ENV_VARS="${ENV_VARS},MIN_CUDA_COMPUTE_CAPABILITY=7.0"
ENV_VARS="${ENV_VARS},MIN_GPU_MEMORY_GB=4.0"

# Legacy compatibility
ENV_VARS="${ENV_VARS},GPU_BATCH_SIZE=8"
ENV_VARS="${ENV_VARS},GPU_MEMORY_LIMIT_GB=20"
ENV_VARS="${ENV_VARS},CHUNK_DURATION_MINUTES=10"
ENV_VARS="${ENV_VARS},ENABLE_BATCH_PROCESSING=true"
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

# External API Keys (from environment)
ENV_VARS="${ENV_VARS},CONVEX_URL=${CONVEX_URL:-}"
ENV_VARS="${ENV_VARS},CONVEX_API_KEY=${CONVEX_API_KEY:-}"
ENV_VARS="${ENV_VARS},DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY:-}"
ENV_VARS="${ENV_VARS},NEBIUS_API_KEY=${NEBIUS_API_KEY:-}"

echo -e "${GREEN}✅ Enhanced GPU optimization configured${NC}"

# Check for required environment variables
echo -e "${BLUE}🔍 Checking required environment variables...${NC}"

required_vars=("CONVEX_URL" "CONVEX_API_KEY" "DEEPGRAM_API_KEY" "NEBIUS_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing environment variables:${NC}"
    printf '   - %s\n' "${missing_vars[@]}"
    echo -e "${YELLOW}   Deployment will continue, but some features may not work${NC}"
else
    echo -e "${GREEN}✅ All required environment variables are set${NC}"
fi

echo ""

# Build Docker image with enhanced GPU support
echo -e "${BLUE}🐳 Building enhanced GPU Docker image...${NC}"

if [ -f "Dockerfile.gpu" ]; then
    echo "Using Dockerfile.gpu for enhanced GPU build..."
    
    # Create temporary cloudbuild.yaml for enhanced GPU build
    cat > cloudbuild.enhanced.yaml << EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-f', 'Dockerfile.gpu', '-t', '${IMAGE_NAME}:enhanced-gpu', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['tag', '${IMAGE_NAME}:enhanced-gpu', '${IMAGE_NAME}:latest']
images:
- '${IMAGE_NAME}:enhanced-gpu'
- '${IMAGE_NAME}:latest'
options:
  machineType: 'E2_HIGHCPU_32'
  diskSizeGb: 100
  logging: CLOUD_LOGGING_ONLY
EOF
    
    gcloud builds submit --config cloudbuild.enhanced.yaml .
    rm cloudbuild.enhanced.yaml
    
elif [ -f "Dockerfile.optimized" ]; then
    echo "Using Dockerfile.optimized for enhanced build..."
    gcloud builds submit --tag ${IMAGE_NAME} -f Dockerfile.optimized .
else
    echo -e "${YELLOW}⚠️  No optimized Dockerfile found, using standard Dockerfile${NC}"
    echo "Note: Enhanced GPU features may not be fully available"
    gcloud builds submit --tag ${IMAGE_NAME} .
fi

echo -e "${GREEN}✅ Enhanced GPU Docker image built and pushed${NC}"

# Verify image exists
echo -e "${BLUE}🔍 Verifying Docker image...${NC}"
if gcloud container images describe ${IMAGE_NAME}:latest --quiet > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker image verified${NC}"
else
    echo -e "${RED}❌ Docker image verification failed${NC}"
    exit 1
fi

echo ""

# Deploy to Cloud Run with enhanced GPU optimization
echo -e "${BLUE}🚀 Deploying to Cloud Run with enhanced GPU optimization...${NC}"
echo "This may take several minutes..."

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --gpu=${GPU_COUNT} \
    --gpu-type=${GPU_TYPE} \
    --cpu ${CPU} \
    --memory ${MEMORY} \
    --timeout ${TIMEOUT}s \
    --allow-unauthenticated \
    --max-instances 3 \
    --min-instances 0 \
    --concurrency 1 \
    --port 8080 \
    --set-env-vars="${ENV_VARS}" \
    --execution-environment gen2 \
    --service-account klipstream-analysis@${PROJECT_ID}.iam.gserviceaccount.com

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Enhanced GPU deployment successful!${NC}"
else
    echo -e "${RED}❌ Enhanced GPU deployment failed${NC}"
    exit 1
fi

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

echo ""
echo -e "${GREEN}🎉 ENHANCED GPU DEPLOYMENT COMPLETED!${NC}"
echo "=" * 60
echo -e "${GREEN}Service URL: ${SERVICE_URL}${NC}"
echo ""

# Test enhanced GPU optimization features
echo -e "${BLUE}🧪 Testing enhanced GPU optimization features...${NC}"

# Test 1: Health check
echo "1️⃣ Health check..."
if curl -s "${SERVICE_URL}/health" > /dev/null; then
    echo -e "${GREEN}   ✅ Health check passed${NC}"
else
    echo -e "${YELLOW}   ⚠️  Health check failed (service may still be starting)${NC}"
fi

# Test 2: Optimization status
echo "2️⃣ Optimization status check..."
if curl -s "${SERVICE_URL}/api/v1/optimization/status" > /dev/null; then
    echo -e "${GREEN}   ✅ Optimization status endpoint available${NC}"
else
    echo -e "${YELLOW}   ⚠️  Optimization status endpoint not available${NC}"
fi

# Test 3: Performance metrics
echo "3️⃣ Performance metrics check..."
if curl -s "${SERVICE_URL}/api/v1/performance/metrics" > /dev/null; then
    echo -e "${GREEN}   ✅ Performance metrics endpoint available${NC}"
else
    echo -e "${YELLOW}   ⚠️  Performance metrics endpoint not available${NC}"
fi

echo ""
echo -e "${GREEN}📋 ENHANCED GPU FEATURES ENABLED:${NC}"
echo "   🔥 Automatic Mixed Precision (AMP)"
echo "   🧠 Advanced Memory Optimization"
echo "   ⚡ Parallel Audio Chunking"
echo "   🎯 Device-Specific Optimizations"
echo "   📊 Comprehensive Performance Monitoring"
echo "   🛡️  Robust Error Handling & Fallbacks"
echo ""

echo -e "${BLUE}📖 MONITORING & DEBUGGING:${NC}"
echo "   • View logs: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit 50"
echo "   • Performance metrics: ${SERVICE_URL}/api/v1/performance/metrics"
echo "   • Optimization status: ${SERVICE_URL}/api/v1/optimization/status"
echo "   • Health check: ${SERVICE_URL}/health"
echo ""

echo -e "${GREEN}🚀 Enhanced GPU-optimized klipstream-analysis is now live!${NC}"
echo -e "${GREEN}   Ready for high-performance transcription workloads${NC}"
