#!/bin/bash
set -e

# Configuration
PROJECT_ID="optimum-habitat-429714-a7"
REGION="us-central1"
SERVICE_NAME="klipstream-analysis"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check if .env.yaml exists
if [ ! -f .env.yaml ]; then
    echo "Error: .env.yaml file not found. Please create it with your API keys."
    exit 1
fi

# Check if service account key exists
if [ ! -f ./new-service-account-key.json ]; then
    echo "Error: Service account key file (new-service-account-key.json) not found."
    echo "Please ensure the service account key file is in the project root directory."
    exit 1
fi

# Determine deployment method
echo "Select deployment method:"
echo "1) Local Docker build (requires Docker installed)"
echo "2) Cloud Build (builds in Google Cloud)"
read -p "Enter your choice (1 or 2): " DEPLOY_METHOD

if [ "$DEPLOY_METHOD" == "1" ]; then
    # Build the Docker image locally
    echo "Building Docker image locally..."
    docker build -t ${IMAGE_NAME} .
    
    # Push the image to Google Container Registry
    echo "Pushing image to Google Container Registry..."
    docker push ${IMAGE_NAME}
elif [ "$DEPLOY_METHOD" == "2" ]; then
    # Build the Docker image using Cloud Build
    echo "Building Docker image using Cloud Build..."
    gcloud builds submit --tag ${IMAGE_NAME} .
else
    echo "Invalid choice. Please enter 1 or 2."
    exit 1
fi

# Configure resource limits
echo "Select resource configuration:"
echo "1) Standard (2 CPU, 4GB memory)"
echo "2) High Memory (2 CPU, 8GB memory)"
echo "3) High CPU (4 CPU, 8GB memory)"
echo "4) Maximum (8 CPU, 32GB memory)"
read -p "Enter your choice (1-4): " RESOURCE_CONFIG

case "$RESOURCE_CONFIG" in
    1)
        CPU="2"
        MEMORY="4Gi"
        TIMEOUT="3600"
        ;;
    2)
        CPU="2"
        MEMORY="8Gi"
        TIMEOUT="3600"
        ;;
    3)
        CPU="4"
        MEMORY="8Gi"
        TIMEOUT="3600"
        ;;
    4)
        CPU="8"
        MEMORY="32Gi"
        TIMEOUT="7200"
        ;;
    *)
        echo "Invalid choice. Using default configuration."
        CPU="2"
        MEMORY="8Gi"
        TIMEOUT="3600"
        ;;
esac

# Update the Cloud Run service
echo "Deploying Cloud Run service with the following configuration:"
echo "- CPU: ${CPU}"
echo "- Memory: ${MEMORY}"
echo "- Timeout: ${TIMEOUT} seconds"
echo "- Service Account: klipstream-service@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --env-vars-file .env.yaml \
    --allow-unauthenticated \
    --service-account="klipstream-service@${PROJECT_ID}.iam.gserviceaccount.com" \
    --cpu=${CPU} \
    --memory=${MEMORY} \
    --timeout=${TIMEOUT}s

# Grant GCS permissions to the service account
echo "Granting GCS permissions to service account..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:klipstream-service@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

echo "Deployment completed!"
echo "Service URL:"
gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)" --project=${PROJECT_ID}
