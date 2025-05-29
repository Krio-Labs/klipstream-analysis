# Google Cloud Storage Authentication

This document explains how to set up authentication for Google Cloud Storage (GCS) in the Klipstream Analysis project.

## Overview

The Klipstream Analysis project uses Google Cloud Storage to store and retrieve various files, including:
- Video files (.mp4)
- Audio files (.wav)
- Waveform files (.png)
- Transcript files (.json, .csv)
- Chat logs (.json, .txt)
- Analysis results (.json, .csv)

These files are stored in the following buckets:
- `klipstream-vods-raw`: For video, audio, and waveform files
- `klipstream-transcripts`: For transcript files
- `klipstream-chatlogs`: For chat log files
- `klipstream-analysis`: For integrated analysis files

## Current Authentication Implementation

The codebase implements a multi-layered authentication approach with fallback mechanisms:

1. **Primary Method**: Service account key file named `new-service-account-key.json` in the project root
2. **Secondary Method**: Service account key file specified by the `GCP_SERVICE_ACCOUNT_PATH` environment variable
3. **Fallback Method**: Application Default Credentials (ADC)

This approach ensures that authentication works in both development and production environments.

### Service Account Authentication

The current implementation first tries to use a service account key file located at `./new-service-account-key.json`:

```python
# Try to use the new service account credentials first
new_key_path = "./new-service-account-key.json"
if os.path.exists(new_key_path):
    try:
        logger.info(f"Using service account credentials from {new_key_path}")
        # Load the service account key file to verify its contents
        with open(new_key_path, 'r') as f:
            key_data = json.load(f)
            logger.info(f"Service account email: {key_data.get('client_email', 'Not found')}")
            logger.info(f"Project ID: {key_data.get('project_id', 'Not found')}")

        credentials = service_account.Credentials.from_service_account_file(
            new_key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        client = storage.Client(credentials=credentials, project=key_data.get('project_id'))
    except Exception as e:
        # Fall back to other authentication methods
        # ...
```

This service account key file is expected to be present in the project root directory and is used by the Cloud Run deployment.

### Fallback Mechanisms

If the primary service account key file is not found or fails to load, the code falls back to the following methods:

1. **GCP_SERVICE_ACCOUNT_PATH**: Checks if this environment variable is set and points to a valid key file
2. **Application Default Credentials**: Uses the credentials provided by `gcloud auth application-default login`

```python
elif GCP_SERVICE_ACCOUNT_PATH and os.path.exists(GCP_SERVICE_ACCOUNT_PATH):
    # Use service account credentials if available
    logger.info(f"Using service account credentials from {GCP_SERVICE_ACCOUNT_PATH}")
    credentials = service_account.Credentials.from_service_account_file(GCP_SERVICE_ACCOUNT_PATH)
    client = storage.Client(credentials=credentials)
else:
    # Use application default credentials
    logger.info("Using application default credentials")
    client = storage.Client()
```

## Setting Up Authentication

### For Cloud Run Deployment

1. **Create a service account** in the Google Cloud Console:
   - Go to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
   - Create a service account named `klipstream-service`
   - Grant it the "Storage Object Admin" role

2. **Create a key** for the service account:
   - Download the key file in JSON format
   - Rename it to `new-service-account-key.json`
   - Place it in the project root directory

3. The deployment script (`deploy_cloud_run_simple.sh`) will automatically:
   - Check for the existence of this file
   - Configure the service account with the necessary permissions
   - Deploy the service with the service account

### For Local Development

You have two options for local development:

#### Option 1: Use the same service account key file

1. Place the `new-service-account-key.json` file in the project root directory
2. The code will automatically use this file for authentication

#### Option 2: Use Application Default Credentials

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Run the following command to authenticate:
   ```bash
   gcloud auth application-default login
   ```
3. Follow the prompts to log in with your Google account
4. Make sure your account has access to the GCS buckets

## Bucket-Level Access Control

The Klipstream GCS buckets are configured with uniform bucket-level access, which means:

1. Access is controlled at the bucket level, not the object level
2. Legacy ACLs are not supported
3. All objects inherit the permissions of the bucket

This configuration requires that the service account has the appropriate permissions at the bucket level.

## Error Handling

The code includes robust error handling for authentication failures:

1. **Validation**: Checks if the service account key file contains valid credentials
2. **Logging**: Provides detailed logs about which authentication method is being used
3. **Fallback**: Automatically falls back to alternative authentication methods if the primary method fails
4. **Graceful Degradation**: Continues execution without GCS integration if all authentication methods fail

## Troubleshooting

If you encounter authentication issues, check the following:

1. **Service Account Key**: Ensure the `new-service-account-key.json` file exists in the project root and contains valid credentials
2. **Permissions**: Verify that the service account has the "Storage Object Admin" role for the buckets
3. **Bucket Names**: Confirm that the bucket names in `utils/config.py` match the actual bucket names in your Google Cloud project
4. **ADC Credentials**: Run `gcloud auth application-default login` to refresh your application default credentials

## Security Best Practices

1. **Never commit service account keys** to version control
2. **Use environment variables** or secure secret management for storing credentials
3. **Limit permissions** of the service account to only what is needed
4. **Rotate service account keys** regularly
5. **Use the principle of least privilege** when granting permissions
6. **Monitor access** to the service account and buckets
