#!/bin/bash
set -e

echo "===== KlipStream Analysis Deployment Script ====="
echo "This script will deploy the KlipStream Analysis service to Google Cloud Run in the Klipstream project."
echo ""

# Configuration
PROJECT_ID="klipstream"
REGION="us-central1"
SERVICE_NAME="klipstream-analysis"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
CPU="8"
MEMORY="32Gi"
TIMEOUT="3600"

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

# Set the project
echo "Setting the project to: ${PROJECT_ID}"
gcloud config set project ${PROJECT_ID}

# Configure Docker to use gcloud as a credential helper
echo "Configuring Docker to use gcloud as a credential helper..."
gcloud auth configure-docker

# Create service account if it doesn't exist
echo "Checking if service account exists..."
SERVICE_ACCOUNT_NAME="klipstream-service"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT_EMAIL} --project=${PROJECT_ID} &>/dev/null; then
    echo "Creating service account..."
    gcloud iam service-accounts create ${SERVICE_ACCOUNT_NAME} \
        --display-name="KlipStream Service Account" \
        --project=${PROJECT_ID}
fi

# Build Docker image with platform specified for AMD64
echo "Building Docker image for AMD64 architecture..."
docker buildx build --platform linux/amd64 -t ${IMAGE_NAME} .

# Push the Docker image to Google Container Registry
echo "Pushing Docker image to Google Container Registry..."
docker push ${IMAGE_NAME}

# Deploy to Cloud Run
echo "Deploying Cloud Run service with the following configuration:"
echo "- CPU: ${CPU}"
echo "- Memory: ${MEMORY}"
echo "- Timeout: ${TIMEOUT} seconds"
echo "- Service Account: ${SERVICE_ACCOUNT_EMAIL}"

gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --update-env-vars="BASE_DIR=/tmp,USE_GCS=true,GCS_PROJECT=${PROJECT_ID},DEEPGRAM_API_KEY=1e5a68cb71082002e20ff76d687e2a2b18806e16,NEBIUS_API_KEY=eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExMDM4NTEzODc2NTM4MjY4NzE0NyIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwNDQwNDY4NSwidXVpZCI6ImM0ZDQzY2IyLWRjMDgtNDc0NS04NzkwLTgxNWMzNzUxYmM2YiIsIm5hbWUiOiJzZW50aW1lbnQiLCJleHBpcmVzX2F0IjoiMjAzMC0wNS0wN1QxNzoxODowNSswMDAwIn0.4s1jXdBv6AZ0tLbKwK2CpdK1zaipgKjCF9CNuwT9I-I" \
  --cpu ${CPU} \
  --memory ${MEMORY} \
  --timeout ${TIMEOUT}s \
  --service-account ${SERVICE_ACCOUNT_EMAIL} \
  --allow-unauthenticated

# Grant GCS permissions to the service account
echo "Granting GCS permissions to service account..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.objectAdmin"

# Grant additional permissions that might be needed
echo "Granting additional permissions to service account..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/cloudfunctions.invoker"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/run.invoker"

# Check if deployment was successful
if [ $? -eq 0 ]; then
  echo "Deployment successful!"

  # Explicitly grant public access to the service
  echo "Configuring public access to the service..."
  gcloud run services add-iam-policy-binding ${SERVICE_NAME} \
    --region=${REGION} \
    --member="allUsers" \
    --role="roles/run.invoker"

  echo "Service URL: $(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')"
else
  echo "Deployment failed. Please check the error message above."
fi
