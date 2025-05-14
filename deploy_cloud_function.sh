#!/bin/bash
set -e

# Configuration
PROJECT_ID="optimum-habitat-429714-a7"
REGION="us-central1"
FUNCTION_NAME="klipstream-analysis"
SERVICE_ACCOUNT="klipstream-service@${PROJECT_ID}.iam.gserviceaccount.com"

# Check if .env.yaml exists
if [ ! -f .env.yaml ]; then
    echo "Error: .env.yaml file not found. Please create it with your API keys."
    exit 1
fi

# Convert .env.yaml to --set-env-vars format
ENV_VARS=$(python3 -c "
import yaml
with open('.env.yaml', 'r') as f:
    env_vars = yaml.safe_load(f)
print(','.join([f'{k}={v}' for k, v in env_vars.items()]))
")

echo "Deploying Cloud Function: ${FUNCTION_NAME}..."
echo "Using environment variables from .env.yaml"

# Deploy the Cloud Function
gcloud functions deploy ${FUNCTION_NAME} \
    --gen2 \
    --runtime=python310 \
    --region=${REGION} \
    --source=. \
    --entry-point=run_pipeline \
    --trigger-http \
    --allow-unauthenticated \
    --service-account=${SERVICE_ACCOUNT} \
    --memory=16GiB \
    --timeout=3600s \
    --set-env-vars=${ENV_VARS}

# Grant GCS permissions to the service account
echo "Granting GCS permissions to service account..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/storage.objectAdmin"

echo "Deployment completed!"
echo "Function URL:"
gcloud functions describe ${FUNCTION_NAME} --region=${REGION} --format="value(serviceConfig.uri)" --project=${PROJECT_ID}

echo ""
echo "Note: Cloud Functions have limitations that may affect the KlipStream Analysis pipeline:"
echo "- Maximum execution time: 60 minutes (3600 seconds)"
echo "- Maximum memory: 16GB"
echo "- Maximum disk space: 10GB in /tmp"
echo ""
echo "For longer or more resource-intensive workloads, consider using Cloud Run instead."
