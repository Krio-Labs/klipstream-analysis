#!/bin/bash
set -e

# Configuration
PROJECT_ID="optimum-habitat-429714-a7"
REGION="us-central1"
SERVICE_NAME="Chat-Audio-Analytics"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check if .env.yaml exists
if [ ! -f .env.yaml ]; then
    echo "Error: .env.yaml file not found. Please create it with your API keys."
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

# Push the image to Google Container Registry
echo "Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}

# Update the Cloud Run service
echo "Updating Cloud Run service..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --env-vars-file .env.yaml \
    --allow-unauthenticated

echo "Deployment completed!"
echo "Service URL:"
gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)" --project=${PROJECT_ID}
