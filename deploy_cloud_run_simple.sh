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
TIMEOUT="3600"  # Maximum allowed by Cloud Run (1 hour)

# Check if .env.yaml exists
if [ ! -f .env.yaml ]; then
    echo "Error: .env.yaml file not found. Please create it with your API keys."
    exit 1
fi

# Load environment variables from .env.yaml
echo "Loading environment variables from .env.yaml..."
eval $(python3 -c "
import yaml
with open('.env.yaml', 'r') as f:
    env_vars = yaml.safe_load(f)
    for key, value in env_vars.items():
        print(f'export {key}=\"{value}\"')
")

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
  --update-env-vars="BASE_DIR=${BASE_DIR},USE_GCS=${USE_GCS},GCS_PROJECT=${GCS_PROJECT},DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY},NEBIUS_API_KEY=${NEBIUS_API_KEY},CONVEX_URL=${CONVEX_URL},CONVEX_API_KEY=${CONVEX_API_KEY},CONVEX_DEPLOYMENT=${CONVEX_DEPLOYMENT},AUTH0_DOMAIN=${AUTH0_DOMAIN},AUTH0_CLIENT_ID=${AUTH0_CLIENT_ID},AUTH0_CLIENT_SECRET=${AUTH0_CLIENT_SECRET}" \
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
