# Cloud Function Deployment

This document outlines the decision-making process, implementation details, and considerations for deploying the KlipStream Analysis pipeline as a Google Cloud Function.

## Background

The KlipStream Analysis pipeline was originally designed to run as a Google Cloud Run service. However, Cloud Functions offer a simpler deployment model with less overhead for certain types of workloads. This document explores the feasibility, benefits, and limitations of migrating to Cloud Functions.

## Current Deployment Status

As of the latest update, the KlipStream Analysis pipeline is primarily deployed as a **Google Cloud Run service** using the `deploy_cloud_run_simple.sh` script. The Cloud Function deployment option is mentioned in documentation but appears to be less actively used, as the `deploy_cloud_function.sh` script is not currently present in the codebase.

## Comparison: Cloud Run vs. Cloud Functions

| Feature | Cloud Run | Cloud Functions (2nd gen) |
|---------|-----------|---------------------------|
| Max Memory | 32GB | 16GB |
| Max CPU | 8 vCPU | 4 vCPU |
| Max Execution Time | 24 hours | 60 minutes |
| Disk Space | Configurable | 10GB (/tmp) |
| Cold Start | Slower | Faster for smaller functions |
| Pricing | Per-second billing | Per-invocation + execution time |
| Container Support | Full Docker support | Limited |
| Concurrency | High | Limited |

## Implementation

### Code Compatibility

The codebase is designed to be compatible with both Cloud Run and Cloud Functions:

1. The `run_pipeline` function in `main.py` is decorated with `@functions_framework.http`, making it compatible with Cloud Functions
2. Environment variables are loaded from `.env.yaml`, which works with both deployment options
3. The Dockerfile includes the Functions Framework as the entry point: `CMD ["functions-framework", "--target=run_pipeline", "--port=8080"]`
4. The code detects the cloud environment using `IS_CLOUD_ENV = os.environ.get('K_SERVICE') is not None`

### Resource Configuration

The current Cloud Run deployment uses the following resources:

```bash
# From deploy_cloud_run_simple.sh
CPU="8"
MEMORY="32Gi"
TIMEOUT="3600"  # 1 hour
```

These settings exceed what's available in Cloud Functions, which is why Cloud Run is the preferred deployment option for production use.

### Environment Configuration

The deployment uses environment variables from `.env.yaml` for configuration, including:

- API keys (Deepgram, Nebius, etc.)
- GCS bucket names
- Project ID

## Limitations and Challenges of Cloud Functions

### Execution Time

The 60-minute maximum execution time of Cloud Functions is insufficient for processing longer VODs. The current Cloud Run deployment uses a 1-hour timeout, which is already at the limit of what Cloud Functions can provide.

### Disk Space

The 10GB `/tmp` storage limit in Cloud Functions may be insufficient for larger VODs. The pipeline uses `/tmp` as the base directory in cloud environments:

```python
# From utils/config.py
BASE_DIR = Path(os.environ.get('BASE_DIR', "/tmp" if IS_CLOUD_ENV else "."))
```

### Memory Constraints

The 16GB memory limit of Cloud Functions is lower than the 32GB configured for the current Cloud Run deployment. This could lead to out-of-memory errors when processing larger files.

### Cold Starts

Cloud Functions that are not frequently invoked experience cold starts, which can add latency to the processing pipeline. This is less of an issue for long-running processes but could impact user experience.

## Current Recommendations

Based on the current codebase and deployment configuration:

1. **Use Cloud Run for Production**: Continue using Cloud Run for production deployments, as it provides the necessary resources (32GB memory, 8 vCPU) for processing VODs of any size.

2. **Consider Cloud Functions for Development**: Cloud Functions could be used for development and testing with smaller VODs, but a deployment script would need to be created.

3. **Optimize for Resource Efficiency**: Continue optimizing the pipeline to reduce memory usage and processing time, which would make Cloud Functions more viable in the future.

4. **Monitor Resource Usage**: Closely monitor memory, disk, and execution time to ensure the pipeline stays within the configured limits.

## Implementation Steps for Cloud Function Deployment

If Cloud Function deployment is desired, the following steps would be needed:

1. Create a `deploy_cloud_function.sh` script that:
   - Reads configuration from `.env.yaml`
   - Deploys a 2nd generation Cloud Function with maximum resources (16GB memory, 4 vCPU)
   - Sets the appropriate environment variables
   - Configures the service account with necessary permissions

2. Optimize the pipeline for the Cloud Function constraints:
   - Implement more aggressive cleanup of temporary files
   - Use streaming processing where possible
   - Upload intermediate results to Cloud Storage more frequently

## Conclusion

While the KlipStream Analysis pipeline is technically compatible with Cloud Functions, Cloud Run remains the recommended deployment option for production use due to its higher resource limits. The current deployment configuration (32GB memory, 8 vCPU) exceeds what Cloud Functions can provide.

For development and testing with smaller VODs, Cloud Functions could be a viable option, but a deployment script would need to be created.
