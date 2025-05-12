# Google Cloud Storage Authentication

This document explains how to set up authentication for Google Cloud Storage (GCS) in the Klipstream Analysis project.

## Overview

The Klipstream Analysis project uses Google Cloud Storage to store raw files, including:
- Video files (.mp4)
- Audio files (.wav)
- Waveform files (.json)
- Transcript files (.csv)
- Chat logs (.csv)

These files are uploaded to the following buckets:
- `klipstream-vods-raw`: For video, audio, and waveform files
- `klipstream-transcripts`: For transcript files
- `klipstream-chatlogs`: For chat log files

## Authentication Options

There are two ways to authenticate with Google Cloud Storage:

1. **Application Default Credentials (ADC)**: This is the simplest method for local development.
2. **Service Account Key**: This is more secure and recommended for production environments.

### Option 1: Application Default Credentials (ADC)

This method uses the gcloud CLI to authenticate with Google Cloud.

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Run the following command to authenticate:
   ```bash
   gcloud auth application-default login
   ```
3. Follow the prompts to log in with your Google account that has access to the GCS buckets.

This will create a credentials file in your home directory that the Google Cloud client libraries will automatically use.

### Option 2: Service Account Key

This method uses a service account key file to authenticate with Google Cloud.

1. Create a service account in the Google Cloud Console:
   - Go to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
   - Click "Create Service Account"
   - Give it a name like "klipstream-analysis"
   - Grant it the "Storage Object Admin" role for the buckets it needs to access

2. Create a key for the service account:
   - Click on the service account you created
   - Go to the "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format
   - Download the key file

3. Set the environment variable to point to the key file:
   - Add the following to your `.env` file:
     ```
     GCP_SERVICE_ACCOUNT_PATH=/path/to/your-service-account-key.json
     ```

## Fallback Behavior

If authentication fails, the pipeline will log a warning and continue without uploading files to GCS. This allows the pipeline to run locally without GCS credentials, which is useful for development and testing.

## Troubleshooting

If you encounter authentication issues, check the following:

1. **ADC Credentials**: Run `gcloud auth application-default login` to refresh your credentials.
2. **Service Account Key**: Ensure the key file exists and the path is correctly specified in the `.env` file.
3. **Permissions**: Ensure the service account or user has the necessary permissions to access the GCS buckets.
4. **Bucket Names**: Verify that the bucket names in `utils/config.py` match the actual bucket names in your Google Cloud project.

## Security Considerations

- Never commit service account keys to version control
- Use environment variables or secure secret management for storing credentials
- Limit the permissions of the service account to only what is needed
- Rotate service account keys regularly
