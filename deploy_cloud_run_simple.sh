#!/bin/bash
set -e

echo "===== KlipStream Analysis Deployment Script (Phase 4) ====="
echo "This script will deploy the FastAPI-based KlipStream Analysis service to Google Cloud Run."
echo "Phase 4: Production deployment with backward compatibility and enhanced features."
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

# Note: Using attached service account instead of key file for better security
echo "Using attached service account for authentication (more secure than key files)"

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

# Deploy to Cloud Run with FastAPI optimizations
echo "Deploying FastAPI-based Cloud Run service with the following configuration:"
echo "- CPU: ${CPU}"
echo "- Memory: ${MEMORY}"
echo "- Timeout: ${TIMEOUT} seconds"
echo "- Service Account: ${SERVICE_ACCOUNT_EMAIL}"
echo "- Framework: FastAPI with uvicorn"
echo "- Features: Async processing, real-time monitoring, queue management"

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
  --allow-unauthenticated \
  --max-instances 10 \
  --concurrency 1000 \
  --port 8080

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

  # Get service URL
  SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
  echo "Service URL: ${SERVICE_URL}"

  # Wait for service to be ready and perform health check
  echo "Waiting for service to be ready..."
  sleep 30

  echo "Performing health check..."
  HEALTH_CHECK_URL="${SERVICE_URL}/health"

  # Try health check with retries
  for i in {1..5}; do
    echo "Health check attempt ${i}/5..."
    if curl -f -s "${HEALTH_CHECK_URL}" > /dev/null; then
      echo "✅ Health check passed!"
      echo "🚀 FastAPI service is running successfully!"
      echo ""
      echo "Available endpoints:"
      echo "  - Health: ${SERVICE_URL}/health"
      echo "  - API Docs: ${SERVICE_URL}/docs"
      echo "  - New API: ${SERVICE_URL}/api/v1/"
      echo "  - Legacy API: ${SERVICE_URL}/legacy/"
      echo "  - Monitoring: ${SERVICE_URL}/api/v1/monitoring/dashboard"
      echo "  - Queue: ${SERVICE_URL}/api/v1/queue/status"
      break
    else
      echo "Health check failed, retrying in 10 seconds..."
      sleep 10
    fi

    if [ $i -eq 5 ]; then
      echo "⚠️  Health check failed after 5 attempts. Service may still be starting up."
      echo "Please check the service logs if issues persist."
    fi
  done

else
  echo "Deployment failed. Please check the error message above."
fi
