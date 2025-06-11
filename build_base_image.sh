#!/bin/bash

# Build base image with all heavy dependencies (run this once)
# This creates a reusable base image for ultra-fast deployments

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ID="klipstream"
BASE_IMAGE_NAME="gcr.io/${PROJECT_ID}/klipstream-base"

echo -e "${BLUE}🏗️  Building KlipStream Base Image${NC}"
echo "=================================="
echo "This will create a reusable base image with all heavy dependencies"
echo "Base image: ${BASE_IMAGE_NAME}"
echo ""

# Create base Dockerfile
cat > Dockerfile.base << 'EOF'
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install all heavy Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.3.0 \
    scipy==1.15.3 \
    scikit-learn==1.7.0 \
    tqdm==4.67.1 \
    python-dotenv==1.1.0 \
    PyYAML==6.0.2 \
    emoji==2.14.1 \
    psutil==7.0.0 \
    ijson==3.4.0 \
    requests==2.32.4 \
    librosa==0.11.0 \
    pydub==0.25.1 \
    soundfile==0.13.1 \
    matplotlib==3.10.3 \
    seaborn==0.13.2 \
    deepgram-sdk==4.3.0 \
    openai==1.86.0 \
    google-cloud-storage==3.1.0 \
    convex==0.7.0 \
    functions-framework==3.8.3 \
    google-api-core==2.25.0 \
    google-auth==2.40.3 \
    aiohttp==3.12.12 \
    fastapi==0.115.12 \
    uvicorn==0.34.3 \
    pydantic==2.11.5

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    torch==2.7.1 \
    torchaudio==2.7.1

# Install Hugging Face and ML models
RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    accelerate==1.7.0

# Install NeMo toolkit (heaviest package)
RUN pip install --no-cache-dir nemo_toolkit[asr]==2.3.1

# Install GPU monitoring tools
RUN pip install --no-cache-dir nvidia-ml-py3 pynvml

# Create app user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Pre-create common directories
RUN mkdir -p /app/output/raw /app/output/analysis /app/output/transcripts /app/output/cost_tracking \
    && mkdir -p /tmp/chunks /tmp/models \
    && chown -R appuser:appuser /app

# Set GPU environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Expose port
EXPOSE 8080
EOF

echo -e "${BLUE}🐳 Building base image...${NC}"
echo "This will take 15-20 minutes but only needs to be done once"

# Build base image using Cloud Build (more resources)
# First create a temporary cloudbuild.yaml for the base image
cat > cloudbuild.base.yaml << 'EOFBUILD'
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-f', 'Dockerfile.base', '-t', '${_IMAGE_NAME}', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', '${_IMAGE_NAME}']
options:
  machineType: 'E2_HIGHCPU_32'
  diskSizeGb: 200
substitutions:
  _IMAGE_NAME: '${BASE_IMAGE_NAME}:latest'
EOFBUILD

gcloud builds submit --config=cloudbuild.base.yaml --substitutions=_IMAGE_NAME=${BASE_IMAGE_NAME}:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Base image built successfully${NC}"
else
    echo -e "${RED}❌ Base image build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Base image pushed to registry automatically${NC}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Base image pushed successfully${NC}"
else
    echo -e "${RED}❌ Base image push failed${NC}"
    exit 1
fi

# Clean up
rm Dockerfile.base

echo ""
echo -e "${GREEN}🎉 Base image ready!${NC}"
echo "Base image: ${BASE_IMAGE_NAME}:latest"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Use deploy_ultra_fast.sh for 1-2 minute deployments"
echo "2. Base image only needs to be rebuilt when dependencies change"
echo "3. Application code changes deploy in ~1-2 minutes"
echo ""

# Create ultra-fast deployment script
cat > deploy_ultra_fast.sh << 'EOF'
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
gcloud builds submit --tag gcr.io/${PROJECT_ID}/klipstream-analysis --file Dockerfile.prebuilt .

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
EOF

chmod +x deploy_ultra_fast.sh

echo -e "${BLUE}💾 Created deploy_ultra_fast.sh for future deployments${NC}"
